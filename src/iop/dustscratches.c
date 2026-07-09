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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "bauhaus/bauhaus.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "develop/imageop_math.h"
#include "common/darktable.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#ifdef HAVE_AI
#include "ai/backend.h"
#include "common/ai_models.h"
#include "control/conf.h"
#endif

#include <gtk/gtk.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

DT_MODULE_INTROSPECTION(3, dt_iop_dustscratches_params_t)

#define DT_DUSTSCRATCHES_MODEL_ID "dust-negative-restore"

typedef struct dt_iop_dustscratches_params_t
{
  float strength; // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0 $DESCRIPTION: "strength"
  float threshold; // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.6 $DESCRIPTION: "mask threshold"
  float radius; // $MIN: 1.0 $MAX: 16.0 $DEFAULT: 4.0 $DESCRIPTION: "repair radius"
  gboolean mark_removed; // $DEFAULT: FALSE $DESCRIPTION: "mark removed dust"
  int run_id; // $DEFAULT: 0
} dt_iop_dustscratches_params_t;

typedef struct dt_iop_dustscratches_gui_data_t
{
  GtkWidget *strength;
  GtkWidget *threshold;
  GtkWidget *radius;
  GtkWidget *mark_removed;
  GtkWidget *run;
} dt_iop_dustscratches_gui_data_t;

typedef struct dt_iop_dustscratches_data_t
{
  float strength;
  float threshold;
  int radius;
  gboolean mark_removed;
  int run_id;
#ifdef HAVE_AI
  GMutex lock;
  dt_ai_environment_t *dev_env;
  dt_ai_context_t *ctx;
  int ctx_width;
  int ctx_height;
  gboolean load_failed;
#endif
} dt_iop_dustscratches_data_t;

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  if(old_version == 1)
  {
    typedef struct dt_iop_dustscratches_params_v1_t
    {
      float strength;
      gboolean mark_removed;
    } dt_iop_dustscratches_params_v1_t;

    const dt_iop_dustscratches_params_v1_t *o = old_params;
    dt_iop_dustscratches_params_t *n = malloc(sizeof(dt_iop_dustscratches_params_t));
    n->strength = o->strength;
    n->threshold = 0.6f;
    n->radius = 4.0f;
    n->mark_removed = o->mark_removed;
    n->run_id = 0;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_dustscratches_params_t);
    *new_version = 3;
    return 0;
  }
  else if(old_version == 2)
  {
    typedef struct dt_iop_dustscratches_params_v2_t
    {
      float strength;
      float threshold;
      float radius;
      gboolean mark_removed;
    } dt_iop_dustscratches_params_v2_t;

    const dt_iop_dustscratches_params_v2_t *o = old_params;
    dt_iop_dustscratches_params_t *n = malloc(sizeof(dt_iop_dustscratches_params_t));
    n->strength = o->strength;
    n->threshold = o->threshold;
    n->radius = o->radius;
    n->mark_removed = o->mark_removed;
    n->run_id = 0;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_dustscratches_params_t);
    *new_version = 3;
    return 0;
  }
  return 1;
}

const char *name()
{
  return _("dust & scratches");
}

const char *aliases()
{
  return _("dust|scratch|negative|film");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self,
                                _("remove dust and scratches from film negatives using an ONNX model"),
                                _("corrective"),
                                _("linear, RGB, scene-referred"),
                                _("linear, RGB"),
                                _("linear, RGB, scene-referred"));
}

int default_group()
{
  return IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL;
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_ONE_INSTANCE;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_dustscratches_params_t *p = (dt_iop_dustscratches_params_t *)p1;
  dt_iop_dustscratches_data_t *d = piece->data;

  d->strength = p->strength;
  d->threshold = p->threshold;
  d->radius = CLAMP((int)lrintf(p->radius), 1, 16);
  d->mark_removed = p->mark_removed;
  d->run_id = p->run_id;
}

static void _run_button_clicked(GtkWidget *widget, dt_iop_module_t *self)
{
  dt_iop_dustscratches_params_t *p = self->params;
  p->run_id++;
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

#ifdef HAVE_AI
static gchar *_dev_model_search_path(void)
{
  const char *candidates[] = {
    "checkpoints",
    "../checkpoints",
    "../../checkpoints",
    NULL
  };
  GString *paths = g_string_new(NULL);

  for(int i = 0; candidates[i]; i++)
  {
    gchar *config_path = g_build_filename(candidates[i],
                                          DT_DUSTSCRATCHES_MODEL_ID,
                                          "config.json",
                                          NULL);
    if(g_file_test(config_path, G_FILE_TEST_EXISTS))
    {
      if(paths->len) g_string_append_c(paths, ';');
      g_string_append(paths, candidates[i]);
    }
    g_free(config_path);
  }

  if(!paths->len)
  {
    g_string_free(paths, TRUE);
    return NULL;
  }

  return g_string_free(paths, FALSE);
}

static dt_ai_environment_t *_get_model_env(dt_iop_dustscratches_data_t *d)
{
  dt_ai_environment_t *env = dt_ai_registry_get_env();
  if(env && dt_ai_get_model_info_by_id(env, DT_DUSTSCRATCHES_MODEL_ID))
    return env;

  if(!d->dev_env)
  {
    gchar *search_path = _dev_model_search_path();
    if(search_path)
    {
      d->dev_env = dt_ai_env_init(search_path);
      g_free(search_path);
    }
  }

  return d->dev_env;
}

static gboolean _ensure_model_loaded(dt_iop_dustscratches_data_t *d,
                                     const int width,
                                     const int height)
{
  if(d->ctx && d->ctx_width == width && d->ctx_height == height)
    return TRUE;

  if(d->ctx)
  {
    dt_ai_unload_model(d->ctx);
    d->ctx = NULL;
  }

  dt_ai_environment_t *env = _get_model_env(d);
  const dt_ai_model_info_t *info = env
    ? dt_ai_get_model_info_by_id(env, DT_DUSTSCRATCHES_MODEL_ID)
    : NULL;

  if(!info)
  {
    if(!d->load_failed)
      dt_print(DT_DEBUG_ALWAYS,
               "[dustscratches] model '%s' not found",
               DT_DUSTSCRATCHES_MODEL_ID);
    d->load_failed = TRUE;
    return FALSE;
  }

  gchar *model_file = dt_ai_model_attribute_string(info, "model_file");
  gchar *height_dim = dt_ai_model_attribute_string(info, "height_dim");
  gchar *width_dim = dt_ai_model_attribute_string(info, "width_dim");

  const dt_ai_dim_override_t dims[] = {
    { height_dim ? height_dim : "height", height },
    { width_dim ? width_dim : "width", width },
  };

  d->ctx = dt_ai_load_model_ext(env,
                                DT_DUSTSCRATCHES_MODEL_ID,
                                model_file ? model_file : "model.onnx",
                                DT_AI_PROVIDER_CONFIGURED,
                                DT_AI_OPT_ALL,
                                dims,
                                2);

  g_free(model_file);
  g_free(height_dim);
  g_free(width_dim);

  if(!d->ctx)
  {
    if(!d->load_failed)
      dt_print(DT_DEBUG_ALWAYS,
               "[dustscratches] failed to load model '%s'",
               DT_DUSTSCRATCHES_MODEL_ID);
    d->load_failed = TRUE;
    return FALSE;
  }

  d->ctx_width = width;
  d->ctx_height = height;
  d->load_failed = FALSE;
  return TRUE;
}
#endif

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  if(!dt_iop_have_required_input_format(4,
                                        self,
                                        piece->colors,
                                        ivoid,
                                        ovoid,
                                        roi_in,
                                        roi_out))
    return;

#ifndef HAVE_AI
  dt_iop_copy_image_roi(ovoid, ivoid, piece->colors, roi_in, roi_out);
  return;
#else
  dt_iop_dustscratches_data_t *d = piece->data;

  if(d->run_id <= 0 || d->strength <= 0.0f || !dt_conf_get_bool("plugins/ai/enabled"))
  {
    dt_iop_copy_image_roi(ovoid, ivoid, piece->colors, roi_in, roi_out);
    return;
  }

  const int width = roi_out->width;
  const int height = roi_out->height;
  const size_t npixels = (size_t)width * height;

  float *mask = dt_alloc_align_float(npixels);
  float *mask_weight = dt_alloc_align_float(npixels);
  if(!mask || !mask_weight)
  {
    dt_free_align(mask);
    dt_free_align(mask_weight);
    dt_iop_copy_image_roi(ovoid, ivoid, piece->colors, roi_in, roi_out);
    return;
  }

  const float *const restrict in = DT_IS_ALIGNED((float *)ivoid);
  float *const restrict out = DT_IS_ALIGNED((float *)ovoid);

  memset(mask, 0, npixels * sizeof(float));
  memset(mask_weight, 0, npixels * sizeof(float));

  gboolean ok = FALSE;
  g_mutex_lock(&d->lock);
  {
    const int tile_size = 512;
    const int overlap = 64;
    const int step = tile_size - 2 * overlap;
    ok = TRUE;

    for(int tile_y = 0; ok && tile_y < height; tile_y += step)
    {
      for(int tile_x = 0; ok && tile_x < width; tile_x += step)
      {
        const int x0 = MIN(tile_x, MAX(0, width - tile_size));
        const int y0 = MIN(tile_y, MAX(0, height - tile_size));
        const int tile_w = MIN(tile_size, width - x0);
        const int tile_h = MIN(tile_size, height - y0);
        const int padded_w = ((tile_w + 15) / 16) * 16;
        const int padded_h = ((tile_h + 15) / 16) * 16;
        const size_t tile_elements = (size_t)padded_w * padded_h;

        float *input = dt_alloc_align_float(tile_elements);
        float *tile_mask = dt_alloc_align_float(tile_elements);
        if(!input || !tile_mask)
        {
          dt_free_align(input);
          dt_free_align(tile_mask);
          ok = FALSE;
          break;
        }

        for(int yy = 0; yy < padded_h; yy++)
        {
          const int src_y = y0 + MIN(yy, tile_h - 1);
          for(int xx = 0; xx < padded_w; xx++)
          {
            const int src_x = x0 + MIN(xx, tile_w - 1);
            const size_t src = (size_t)src_y * width + src_x;
            input[(size_t)yy * padded_w + xx]
              = CLIP(0.2126f * in[4 * src]
                     + 0.7152f * in[4 * src + 1]
                     + 0.0722f * in[4 * src + 2]);
          }
        }

        int64_t tensor_shape[] = { 1, 1, padded_h, padded_w };
        dt_ai_tensor_t inputs[] = {
          { input, DT_AI_FLOAT, tensor_shape, 4 },
        };
        dt_ai_tensor_t outputs[] = {
          { tile_mask, DT_AI_FLOAT, tensor_shape, 4 },
        };

        if(!_ensure_model_loaded(d, padded_w, padded_h)
           || dt_ai_run(d->ctx, inputs, 1, outputs, 1) != 0)
        {
          ok = FALSE;
        }
        else
        {
          for(int yy = 0; yy < tile_h; yy++)
          {
            for(int xx = 0; xx < tile_w; xx++)
            {
              const size_t dst = (size_t)(y0 + yy) * width + x0 + xx;
              mask[dst] += tile_mask[(size_t)yy * padded_w + xx];
              mask_weight[dst] += 1.0f;
            }
          }
        }

        dt_free_align(input);
        dt_free_align(tile_mask);
      }
    }
  }
  g_mutex_unlock(&d->lock);

  if(ok)
  {
    DT_OMP_FOR()
    for(size_t k = 0; k < npixels; k++)
      mask[k] = mask_weight[k] > 0.0f ? mask[k] / mask_weight[k] : 0.0f;

    const float strength = d->strength;
    const float threshold = d->threshold;
    const float inv_range = (threshold < 1.0f) ? 1.0f / (1.0f - threshold) : 1.0f;
    const int radius = d->radius;

    DT_OMP_FOR()
    for(int y = 0; y < height; y++)
    {
      for(int x = 0; x < width; x++)
      {
        const size_t k = (size_t)y * width + x;
        const float probability = CLIP(mask[k]);
        const float detected = CLIP((probability - threshold) * inv_range);
        const float blend = strength * detected;

        if(d->mark_removed && blend > 0.0f)
        {
          out[4 * k] = 1.0f;
          out[4 * k + 1] = 0.0f;
          out[4 * k + 2] = 0.0f;
          out[4 * k + 3] = in[4 * k + 3];
          continue;
        }

        if(blend <= 0.0f || x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1)
        {
          for_four_channels(c)
            out[4 * k + c] = in[4 * k + c];
          continue;
        }

        float repair[3] = { 0.0f };
        float weight = 0.0f;
        const int ymin = MAX(0, y - radius);
        const int ymax = MIN(height - 1, y + radius);
        const int xmin = MAX(0, x - radius);
        const int xmax = MIN(width - 1, x + radius);

        for(int yy = ymin; yy <= ymax; yy++)
        {
          for(int xx = xmin; xx <= xmax; xx++)
          {
            if(xx == x && yy == y) continue;
            const size_t kk = (size_t)yy * width + xx;
            const float neighbor_probability = CLIP(mask[kk]);
            const float neighbor_detected = CLIP((neighbor_probability - threshold) * inv_range);
            const float distance2 = (float)((xx - x) * (xx - x) + (yy - y) * (yy - y));
            const float w = (1.0f - neighbor_detected) / (1.0f + distance2);
            for(int c = 0; c < 3; c++)
              repair[c] += w * in[4 * kk + c];
            weight += w;
          }
        }

        if(weight <= 1e-6f)
        {
          for(int c = 0; c < 3; c++)
            repair[c] = in[4 * k + c];
        }
        else
        {
          for(int c = 0; c < 3; c++)
            repair[c] /= weight;
        }

        for(int c = 0; c < 3; c++)
          out[4 * k + c] = (1.0f - blend) * in[4 * k + c] + blend * repair[c];
        out[4 * k + 3] = in[4 * k + 3];
      }
    }
  }
  else
  {
    dt_iop_copy_image_roi(ovoid, ivoid, piece->colors, roi_in, roi_out);
  }

  dt_free_align(mask);
  dt_free_align(mask_weight);
#endif
}

void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_dustscratches_data_t *d = calloc(1, sizeof(dt_iop_dustscratches_data_t));
  if(!d) return;
#ifdef HAVE_AI
  g_mutex_init(&d->lock);
#endif
  piece->data = d;
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_dustscratches_data_t *d = piece->data;
#ifdef HAVE_AI
  if(d)
  {
    if(d->ctx)
      dt_ai_unload_model(d->ctx);
    if(d->dev_env)
      dt_ai_env_destroy(d->dev_env);
    g_mutex_clear(&d->lock);
  }
#endif
  free(d);
  piece->data = NULL;
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_dustscratches_gui_data_t *g = IOP_GUI_ALLOC(dustscratches);

  g->strength = dt_bauhaus_slider_from_params(self, N_("strength"));
  dt_bauhaus_slider_set_format(g->strength, "%");
  gtk_widget_set_tooltip_text(g->strength, _("strength of mask-guided dust and scratch repair"));

  g->threshold = dt_bauhaus_slider_from_params(self, "threshold");
  dt_bauhaus_slider_set_format(g->threshold, "%");
  gtk_widget_set_tooltip_text(g->threshold, _("lower values repair weaker detections"));

  g->radius = dt_bauhaus_slider_from_params(self, "radius");
  dt_bauhaus_slider_set_format(g->radius, _(" px"));
  gtk_widget_set_tooltip_text(g->radius, _("neighborhood size used to fill detected defects"));

  g->mark_removed = dt_bauhaus_toggle_from_params(self, "mark_removed");
  gtk_widget_set_tooltip_text(g->mark_removed, _("show detected dust and scratches in red"));

  g->run = gtk_button_new_with_label(_("run dust removal"));
  gtk_widget_set_tooltip_text(g->run, _("run ONNX dust and scratch removal with the current settings"));
  g_signal_connect(G_OBJECT(g->run), "clicked", G_CALLBACK(_run_button_clicked), self);
  dt_gui_box_add(self->widget, g->run);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
