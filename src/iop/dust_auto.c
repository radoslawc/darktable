/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "bauhaus/bauhaus.h"
#include "common/box_filters.h"
#include "common/imagebuf.h"
#include "common/math.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DUST_AUTO_MAX_RADIUS 16
#define DUST_AUTO_HASH_SCALE 0.1031f

DT_MODULE_INTROSPECTION(2, dt_iop_dust_auto_params_t)

typedef struct dt_iop_dust_auto_params_t
{
  float threshold; // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.66
  float auto_size; // $MIN: 1.0 $MAX: 16.0 $DEFAULT: 4.0 $DESCRIPTION: "Auto size"
  gboolean auto_parameters; // $DEFAULT: FALSE $DESCRIPTION: "Auto parameters"
  gboolean mark_removed; // $DEFAULT: FALSE $DESCRIPTION: "Mark removed spots"
} dt_iop_dust_auto_params_t;

typedef struct dt_iop_dust_auto_gui_data_t
{
  GtkWidget *threshold, *auto_size;
  GtkToggleButton *auto_parameters;
  GtkToggleButton *mark_removed;
  GtkLabel *message;
  float selected_threshold;
  float selected_auto_size;
  gboolean auto_parameters_used;
  int spots_removed;
} dt_iop_dust_auto_gui_data_t;

typedef struct dt_iop_dust_auto_data_t
{
  float threshold;
  float auto_size;
  gboolean auto_parameters;
  gboolean mark_removed;
} dt_iop_dust_auto_data_t;


const char *name()
{
  return _("Auto dust removal");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("automatically remove dust spots"),
                                      _("corrective"),
                                      _("linear, RGB, scene-referred"),
                                      _("reconstruction, RGB"),
                                      _("linear, RGB, scene-referred"));
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

int default_group()
{
  return IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  if(old_version == 1)
  {
    typedef struct dt_iop_dust_auto_params_v1_t
    {
      float threshold;
      float auto_size;
      gboolean mark_removed;
    } dt_iop_dust_auto_params_v1_t;

    const dt_iop_dust_auto_params_v1_t *o = old_params;
    dt_iop_dust_auto_params_t *n = malloc(sizeof(dt_iop_dust_auto_params_t));

    n->threshold = o->threshold;
    n->auto_size = o->auto_size;
    n->auto_parameters = FALSE;
    n->mark_removed = o->mark_removed;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_dust_auto_params_t);
    *new_version = 2;
    return 0;
  }

  return 1;
}

static inline float _luma(const float *const pixel)
{
  return 0.2126f * pixel[0] + 0.7152f * pixel[1] + 0.0722f * pixel[2];
}

static inline float _romm_encode(const float x)
{
  const float v = CLAMP(x, 0.0f, 1.0f);
  return (v < 1.0f / 512.0f) ? v * 16.0f : powf(v, 1.0f / 1.8f);
}

static inline int _clampi(const int v, const int low, const int high)
{
  return MIN(MAX(v, low), high);
}

static inline float _fractf(const float x)
{
  return x - floorf(x);
}

static inline float _hash2(const int x, const int y)
{
  float px = _fractf((float)x * DUST_AUTO_HASH_SCALE);
  float py = _fractf((float)y * DUST_AUTO_HASH_SCALE);
  float pz = px;
  const float d = px * (py + 33.33f) + py * (pz + 33.33f) + pz * (px + 33.33f);
  px += d;
  py += d;
  pz += d;
  return _fractf((px + py) * pz);
}

static void _box_mean(const float *const in,
                      float *const out,
                      const int width,
                      const int height,
                      const int radius)
{
  if(out != in) memcpy(out, in, sizeof(float) * (size_t)width * height);
  dt_box_mean(out, height, width, 1, radius, 1);
}

static void _build_detection_stats(const float *const restrict in,
                                   float *const restrict gray,
                                   float *const restrict gray2,
                                   float *const restrict mean,
                                   float *const restrict local_std,
                                   float *const restrict wide_std,
                                   const int width,
                                   const int height,
                                   const float auto_size,
                                   const float scale)
{
  const size_t npixels = (size_t)width * height;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float *const pixel = in + 4 * k;
    const dt_aligned_pixel_t enc = { _romm_encode(pixel[0]),
                                     _romm_encode(pixel[1]),
                                     _romm_encode(pixel[2]),
                                     pixel[3] };
    gray[k] = _luma(enc);
    gray2[k] = gray[k] * gray[k];
  }

  const int local_radius = (int)MAX(3.0f, auto_size * 3.0f * scale);
  _box_mean(gray, mean, width, height, local_radius);
  _box_mean(gray2, local_std, width, height, local_radius);

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
    local_std[k] = sqrtf(MAX(0.0f, local_std[k] - mean[k] * mean[k]));

  const int wide_radius = (int)MAX(7.0f, auto_size * 4.0f * scale);
  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
    gray2[k] = gray[k] * gray[k];

  _box_mean(gray, wide_std, width, height, wide_radius);
  _box_mean(gray2, gray2, width, height, wide_radius);

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
    wide_std[k] = sqrtf(MAX(0.0f, gray2[k] - wide_std[k] * wide_std[k]));
}

static int _detect_dust(const float *const restrict gray,
                        const float *const restrict mean,
                        const float *const restrict local_std,
                        const float *const restrict wide_std,
                        uint8_t *const restrict hit_mask,
                        const int width,
                        const int height,
                        const float threshold)
{
  int hits = 0;

  DT_OMP_FOR(reduction(+:hits))
  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      const size_t pos = (size_t)y * width + x;
      const float l_curr = gray[pos];
      const float l_mean = mean[pos];
      const float local_s = MAX(0.005f, local_std[pos]);
      const float w_s = MAX(0.0f, wide_std[pos] - 0.02f);
      const float wide_penalty = w_s * w_s * w_s * 800.0f;
      const float thresh = (threshold * 0.4f) + local_s + wide_penalty;
      const float diff = l_curr - l_mean;

      if(diff > thresh && l_curr > 0.15f && diff / local_s > 3.0f)
      {
        const gboolean is_strong = diff > thresh * 2.5f || diff > 0.25f;
        gboolean is_max = TRUE;

        if(x > 0 && y > 0 && x < width - 1 && y < height - 1)
        {
          for(int dy = -1; dy <= 1 && is_max; dy++)
          {
            for(int dx = -1; dx <= 1; dx++)
            {
              if(dx == 0 && dy == 0) continue;
              const float neighbour = gray[(size_t)(y + dy) * width + x + dx];
              if(neighbour >= l_curr)
              {
                is_max = FALSE;
                break;
              }
            }
          }
        }
        else
        {
          is_max = TRUE;
        }

        if(is_max || is_strong)
        {
          hit_mask[pos] = 1;
          hits++;
        }
      }
    }
  }

  return hits;
}

static int _detect_hits(const float *const restrict in,
                        float *const restrict gray,
                        float *const restrict gray2,
                        float *const restrict mean,
                        float *const restrict local_std,
                        float *const restrict wide_std,
                        uint8_t *const restrict hit_mask,
                        const int width,
                        const int height,
                        const float threshold,
                        const float auto_size,
                        const float scale)
{
  memset(hit_mask, 0, sizeof(uint8_t) * (size_t)width * height);
  _build_detection_stats(in, gray, gray2, mean, local_std, wide_std,
                         width, height, auto_size, scale);
  return _detect_dust(gray, mean, local_std, wide_std, hit_mask, width, height, threshold);
}

static int _auto_detect_hits(const float *const restrict in,
                             float *const restrict gray,
                             float *const restrict gray2,
                             float *const restrict mean,
                             float *const restrict local_std,
                             float *const restrict wide_std,
                             uint8_t *const restrict hit_mask,
                             const int width,
                             const int height,
                             const float scale,
                             float *const restrict selected_threshold,
                             float *const restrict selected_auto_size)
{
  const float thresholds[] = { 0.85f, 0.72f, 0.60f, 0.48f, 0.36f, 0.26f, 0.18f };
  const float sizes[] = { 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f };
  const size_t npixels = (size_t)width * height;
  const int preferred_max = MAX(32, (int)(npixels / 1000));
  float best_threshold = thresholds[0];
  float best_auto_size = sizes[0];
  int best_spots_count = -1;
  int fallback_spots_count = -1;
  float fallback_threshold = thresholds[0];
  float fallback_auto_size = sizes[0];

  for(size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++)
  {
    for(size_t t = 0; t < sizeof(thresholds) / sizeof(thresholds[0]); t++)
    {
      const int spots_count = _detect_hits(in, gray, gray2, mean, local_std, wide_std,
                                           hit_mask, width, height, thresholds[t], sizes[s], scale);

      if(spots_count > fallback_spots_count)
      {
        fallback_spots_count = spots_count;
        fallback_threshold = thresholds[t];
        fallback_auto_size = sizes[s];
      }

      if(spots_count <= preferred_max && spots_count > best_spots_count)
      {
        best_spots_count = spots_count;
        best_threshold = thresholds[t];
        best_auto_size = sizes[s];
      }
    }
  }

  if(best_spots_count < 0)
  {
    best_spots_count = fallback_spots_count;
    best_threshold = fallback_threshold;
    best_auto_size = fallback_auto_size;
  }

  *selected_threshold = best_threshold;
  *selected_auto_size = best_auto_size;

  return _detect_hits(in, gray, gray2, mean, local_std, wide_std,
                      hit_mask, width, height, best_threshold, best_auto_size, scale);
}

static void _sort_luma_samples(float *const restrict luma,
                               float *const restrict red,
                               float *const restrict green,
                               float *const restrict blue)
{
  for(int i = 0; i < 8; i++)
  {
    for(int j = i + 1; j < 8; j++)
    {
      if(luma[i] > luma[j])
      {
        const float tl = luma[i];
        const float tr = red[i];
        const float tg = green[i];
        const float tb = blue[i];
        luma[i] = luma[j];
        red[i] = red[j];
        green[i] = green[j];
        blue[i] = blue[j];
        luma[j] = tl;
        red[j] = tr;
        green[j] = tg;
        blue[j] = tb;
      }
    }
  }
}

static void _prepare_heal_mask(const uint8_t *const restrict hit_mask,
                               float *const restrict nearest_d2,
                               float *const restrict center_x,
                               float *const restrict center_y,
                               float *const restrict center_count,
                               const int width,
                               const int height,
                               const int exp_rad)
{
  const size_t npixels = (size_t)width * height;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    nearest_d2[k] = 1.0e9f;
    center_x[k] = 0.0f;
    center_y[k] = 0.0f;
    center_count[k] = 0.0f;
  }

  for(int hy = 0; hy < height; hy++)
  {
    for(int hx = 0; hx < width; hx++)
    {
      if(!hit_mask[(size_t)hy * width + hx]) continue;

      const int min_y = _clampi(hy - exp_rad, 0, height - 1);
      const int max_y = _clampi(hy + exp_rad, 0, height - 1);
      const int min_x = _clampi(hx - exp_rad, 0, width - 1);
      const int max_x = _clampi(hx + exp_rad, 0, width - 1);

      for(int y = min_y; y <= max_y; y++)
      {
        const int dy = hy - y;
        for(int x = min_x; x <= max_x; x++)
        {
          const int dx = hx - x;
          const float d2 = (float)(dx * dx + dy * dy);
          const size_t pos = (size_t)y * width + x;

          nearest_d2[pos] = MIN(nearest_d2[pos], d2);
          center_x[pos] += (float)hx;
          center_y[pos] += (float)hy;
          center_count[pos] += 1.0f;
        }
      }
    }
  }
}

static gboolean _fallback_heal_sample(const float *const restrict in,
                                      dt_aligned_pixel_t healed,
                                      const int x,
                                      const int y,
                                      const int width,
                                      const int height,
                                      const int p_rad)
{
  const int dxs[8] = { 0, 0, -p_rad, p_rad, -p_rad, p_rad, -p_rad, p_rad };
  const int dys[8] = { -p_rad, p_rad, 0, 0, -p_rad, -p_rad, p_rad, p_rad };
  float red[8], green[8], blue[8], luma[8];

  for(int i = 0; i < 8; i++)
  {
    const int sy = _clampi(y + dys[i], 0, height - 1);
    const int sx = _clampi(x + dxs[i], 0, width - 1);
    const float *const pixel = in + 4 * ((size_t)sy * width + sx);
    red[i] = pixel[0];
    green[i] = pixel[1];
    blue[i] = pixel[2];
    luma[i] = _luma(pixel);
  }

  _sort_luma_samples(luma, red, green, blue);

  healed[0] = (red[2] + red[3] + red[4] + red[5]) * 0.25f;
  healed[1] = (green[2] + green[3] + green[4] + green[5]) * 0.25f;
  healed[2] = (blue[2] + blue[3] + blue[4] + blue[5]) * 0.25f;
  healed[3] = in[4 * ((size_t)y * width + x) + 3];

  return TRUE;
}

static void _heal_dust(const float *const restrict in,
                       float *const restrict out,
                       const uint8_t *const restrict hit_mask,
                       const int hit_count,
                       float *const restrict nearest_d2,
                       float *const restrict center_x,
                       float *const restrict center_y,
                       float *const restrict center_count,
                       const int width,
                       const int height,
                       const float auto_size,
                       const float scale,
                       const gboolean mark_removed)
{
  if(hit_count <= 0) return;

  const int exp_rad = _clampi((int)MAX(1.0f, auto_size * 0.4f * scale),
                              1, DUST_AUTO_MAX_RADIUS);
  const int p_rad = exp_rad + (int)(3.0f * scale);
  const float cos_a[5] = { 1.0f, 0.7071067811865476f, 0.7071067811865476f, 0.0f, 0.0f };
  const float sin_a[5] = { 0.0f, 0.7071067811865476f, -0.7071067811865476f, 1.0f, -1.0f };
  const float heal_limit = (float)(exp_rad * exp_rad + 1);

  _prepare_heal_mask(hit_mask, nearest_d2, center_x, center_y, center_count,
                     width, height, exp_rad);

  DT_OMP_FOR()
  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      const size_t ipos = (size_t)y * width + x;
      const float count = center_count[ipos];
      if(count <= 0.0f || nearest_d2[ipos] >= heal_limit) continue;

      float feather = 1.0f - sqrtf(nearest_d2[ipos]) / (float)(exp_rad + 1);
      feather = CLAMP(feather, 0.0f, 1.0f);
      feather = feather * feather * (3.0f - 2.0f * feather);
      if(feather <= 0.001f) continue;

      const size_t opos = 4 * ipos;

      if(mark_removed)
      {
        out[opos] = 1.0f;
        out[opos + 1] = 0.0f;
        out[opos + 2] = 0.0f;
        continue;
      }

      float ux = (float)x - center_x[ipos] / count;
      float uy = (float)y - center_y[ipos] / count;
      const float ul = sqrtf(ux * ux + uy * uy);

      if(ul < 1.0e-3f)
      {
        const float ang = _hash2(x, y) * 6.28318530718f;
        ux = cosf(ang);
        uy = sinf(ang);
      }
      else
      {
        ux /= ul;
        uy /= ul;
      }

      dt_aligned_pixel_t healed = { 0.0f };
      gboolean found = FALSE;

      for(int k = 0; k < 5; k++)
      {
        const float rx_dir = ux * cos_a[k] - uy * sin_a[k];
        const float ry_dir = ux * sin_a[k] + uy * cos_a[k];
        const int sx = _clampi((int)roundf((float)x + rx_dir * (float)p_rad), 0, width - 1);
        const int sy = _clampi((int)roundf((float)y + ry_dir * (float)p_rad), 0, height - 1);

        if(!hit_mask[(size_t)sy * width + sx])
        {
          const float *const source = in + 4 * ((size_t)sy * width + sx);
          for(int c = 0; c < 3; c++)
            healed[c] = source[c];
          healed[3] = in[opos + 3];
          found = TRUE;
          break;
        }
      }

      if(!found)
        _fallback_heal_sample(in, healed, x, y, width, height, p_rad);

      for(int c = 0; c < 3; c++)
        out[opos + c] = in[opos + c] * (1.0f - feather) + healed[c] * feather;
    }
  }
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  if(!dt_iop_have_required_input_format(4 /*we need full-color pixels*/,
                                        self, piece->colors,
                                        ivoid, ovoid, roi_in, roi_out))
    return;

  const dt_iop_dust_auto_data_t *const d = piece->data;
  dt_iop_dust_auto_gui_data_t *const g = self->gui_data;
  const float *const restrict in = (const float *)ivoid;
  float *const restrict out = (float *)ovoid;
  const int width = roi_out->width;
  const int height = roi_out->height;
  const size_t npixels = (size_t)width * height;

  dt_iop_image_copy_by_size(ovoid, ivoid, roi_out->width, roi_out->height, 4);

  float *const restrict gray = dt_alloc_align_float(npixels);
  float *const restrict gray2 = dt_alloc_align_float(npixels);
  float *const restrict mean = dt_alloc_align_float(npixels);
  float *const restrict local_std = dt_alloc_align_float(npixels);
  float *const restrict wide_std = dt_alloc_align_float(npixels);
  uint8_t *const restrict hit_mask = calloc(npixels, sizeof(uint8_t));
  int hit_count = 0;
  float threshold = CLAMP(d->threshold, 0.0f, 1.0f);
  float auto_size = MAX(1.0f, d->auto_size);

  if(gray && gray2 && mean && local_std && wide_std && hit_mask)
  {
    const float scale = MAX(1.0f, fmaxf(width, height) / 1024.0f);

    if(d->auto_parameters)
      hit_count = _auto_detect_hits(in, gray, gray2, mean, local_std, wide_std,
                                    hit_mask, width, height, scale, &threshold, &auto_size);
    else
      hit_count = _detect_hits(in, gray, gray2, mean, local_std, wide_std,
                               hit_mask, width, height, threshold, auto_size, scale);

    _heal_dust(in, out, hit_mask, hit_count, mean, local_std, wide_std, gray2,
               width, height, auto_size, scale, d->mark_removed);
  }

  if(g && self->dev->gui_attached && dt_pipe_is_full(piece->pipe))
  {
    g->selected_threshold = threshold;
    g->selected_auto_size = auto_size;
    g->auto_parameters_used = d->auto_parameters;
    g->spots_removed = hit_count;
  }

  dt_free_align(gray);
  dt_free_align(gray2);
  dt_free_align(mean);
  dt_free_align(local_std);
  dt_free_align(wide_std);
  free(hit_mask);
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_dust_auto_params_t *p = (dt_iop_dust_auto_params_t *)p1;
  dt_iop_dust_auto_data_t *d = piece->data;

  d->threshold = p->threshold;
  d->auto_size = p->auto_size;
  d->auto_parameters = p->auto_parameters;
  d->mark_removed = p->mark_removed;
}

void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_dust_auto_data_t));
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_dust_auto_gui_data_t *g = self->gui_data;
  g->spots_removed = -1;
  g->auto_parameters_used = FALSE;
  gtk_label_set_text(g->message, "");
}

static gboolean draw(GtkWidget *widget, cairo_t *cr, dt_iop_module_t *self)
{
  dt_iop_dust_auto_gui_data_t *g = self->gui_data;
  DT_GUARD_GUI_UPDATE(FALSE);

  if(g->spots_removed < 0) return FALSE;

  char *str = NULL;
  if(g->auto_parameters_used)
    str = g_strdup_printf(ngettext("removed %d spot (threshold %.2f, size %.1f)",
                                   "removed %d spots (threshold %.2f, size %.1f)",
                                   g->spots_removed),
                          g->spots_removed, g->selected_threshold, g->selected_auto_size);
  else
    str = g_strdup_printf(ngettext("removed %d spot", "removed %d spots", g->spots_removed),
                          g->spots_removed);
  g->spots_removed = -1;

  DT_ENTER_GUI_UPDATE();
  gtk_label_set_text(g->message, str);
  DT_LEAVE_GUI_UPDATE();

  g_free(str);

  return FALSE;
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_dust_auto_gui_data_t *g = IOP_GUI_ALLOC(dust_auto);

  g->spots_removed = -1;
  g->selected_threshold = 0.0f;
  g->selected_auto_size = 0.0f;
  g->auto_parameters_used = FALSE;
  g_signal_connect(G_OBJECT(self->widget), "draw", G_CALLBACK(draw), self);

  g->threshold = dt_bauhaus_slider_from_params(self, N_("threshold"));
  dt_bauhaus_slider_set_digits(g->threshold, 3);
  gtk_widget_set_tooltip_text(g->threshold, _("threshold for automatic dust detection"));

  g->auto_size = dt_bauhaus_slider_from_params(self, "auto_size");
  dt_bauhaus_slider_set_digits(g->auto_size, 1);
  gtk_widget_set_tooltip_text(g->auto_size, _("size of automatically detected dust spots"));

  g->auto_parameters = GTK_TOGGLE_BUTTON(dt_bauhaus_toggle_from_params(self, "auto_parameters"));
  gtk_widget_set_tooltip_text(GTK_WIDGET(g->auto_parameters),
                              _("automatically choose threshold and size"));

  g->mark_removed = GTK_TOGGLE_BUTTON(dt_bauhaus_toggle_from_params(self, "mark_removed"));
  gtk_widget_set_tooltip_text(GTK_WIDGET(g->mark_removed), _("mark removed dust spots in red"));

  GtkWidget *hbox = dt_gui_hbox();
  g->message = GTK_LABEL(gtk_label_new(""));
  dt_gui_box_add(hbox, g->message);
  dt_gui_box_add(self->widget, hbox);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
