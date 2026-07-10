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
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <float.h>
#include <gtk/gtk.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

DT_MODULE_INTROSPECTION(1, dt_iop_autoinputlevels_params_t)

#define AUTO_INPUT_LEVELS_BINS 1024

typedef struct dt_iop_autoinputlevels_params_t
{
  float clipping; // $MIN: 0.0 $MAX: 10.0 $DEFAULT: 0.6 $DESCRIPTION: "clipping"
} dt_iop_autoinputlevels_params_t;

typedef struct dt_iop_autoinputlevels_gui_data_t
{
  GtkWidget *clipping;
} dt_iop_autoinputlevels_gui_data_t;

typedef struct dt_iop_autoinputlevels_data_t
{
  float clipping;
} dt_iop_autoinputlevels_data_t;

const char *name()
{
  return _("auto input levels");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("automatically stretch RGB input levels"),
                                      _("corrective"),
                                      _("linear, RGB, display-referred"),
                                      _("linear, RGB"),
                                      _("linear, RGB, display-referred"));
}

int default_group()
{
  return IOP_GROUP_TONE | IOP_GROUP_GRADING;
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

static void _find_channel_extrema(const uint64_t *const hist,
                                  float *const low,
                                  float *const high)
{
  *low = 0.0f;
  *high = 1.0f;

  for(int i = 0; i < AUTO_INPUT_LEVELS_BINS; i++)
  {
    if(hist[i])
    {
      *low = (float)i / (float)(AUTO_INPUT_LEVELS_BINS - 1);
      break;
    }
  }

  for(int i = AUTO_INPUT_LEVELS_BINS - 1; i >= 0; i--)
  {
    if(hist[i])
    {
      *high = (float)i / (float)(AUTO_INPUT_LEVELS_BINS - 1);
      break;
    }
  }
}

static void _stretch_channel(const uint64_t *const hist,
                             const uint64_t count,
                             const float clipping,
                             float *const low,
                             float *const high)
{
  *low = 0.0f;
  *high = 1.0f;

  if(count == 0) return;

  if(clipping <= 0.0f)
  {
    _find_channel_extrema(hist, low, high);
  }
  else
  {
    const double bias = (double)clipping / 100.0;
    const double total = (double)count;
    uint64_t new_count = 0;

    for(int i = 0; i < AUTO_INPUT_LEVELS_BINS - 1; i++)
    {
      new_count += hist[i];
      const double percentage = (double)new_count / total;
      const double next_percentage = (double)(new_count + hist[i + 1]) / total;

      if(fabs(percentage - bias) < fabs(next_percentage - bias))
      {
        *low = (float)(i + 1) / (float)(AUTO_INPUT_LEVELS_BINS - 1);
        break;
      }
    }

    new_count = 0;

    for(int i = AUTO_INPUT_LEVELS_BINS - 1; i > 0; i--)
    {
      new_count += hist[i];
      const double percentage = (double)new_count / total;
      const double next_percentage = (double)(new_count + hist[i - 1]) / total;

      if(fabs(percentage - bias) < fabs(next_percentage - bias))
      {
        *high = (float)(i - 1) / (float)(AUTO_INPUT_LEVELS_BINS - 1);
        break;
      }
    }
  }

  if(*high - *low <= FLT_EPSILON)
  {
    *low = 0.0f;
    *high = 1.0f;
  }
}

static void _compute_levels(const float *const img,
                            const size_t npixels,
                            const float clipping,
                            float low[3],
                            float high[3])
{
  uint64_t histogram[3][AUTO_INPUT_LEVELS_BINS] = { 0 };
  uint64_t count[3] = { 0 };

  for(size_t k = 0; k < npixels; k++)
  {
    const float *const pixel = img + 4 * k;

    for(int c = 0; c < 3; c++)
    {
      if(isfinite(pixel[c]))
      {
        const float value = CLAMP(pixel[c], 0.0f, 1.0f);
        const int bin = CLAMP((int)(value * (AUTO_INPUT_LEVELS_BINS - 1) + 0.5f),
                              0, AUTO_INPUT_LEVELS_BINS - 1);
        histogram[c][bin]++;
        count[c]++;
      }
    }
  }

  for(int c = 0; c < 3; c++)
  {
    _stretch_channel(histogram[c], count[c], clipping, low + c, high + c);
  }
}

static inline float _map_input_level(const float value,
                                     const float low,
                                     const float high)
{
  if(value <= low) return 0.0f;

  const float range = high - low;
  return (range > FLT_EPSILON) ? (value - low) / range : value;
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  if(!dt_iop_have_required_input_format(4, self, piece->colors, ivoid, ovoid, roi_in, roi_out))
    return;

  const dt_iop_autoinputlevels_data_t *const data = piece->data;
  const float *const restrict in = (const float *)ivoid;
  float *const restrict out = (float *)ovoid;
  const size_t npixels = (size_t)roi_out->width * roi_out->height;

  float low[3] = { 0.0f };
  float high[3] = { 1.0f, 1.0f, 1.0f };
  _compute_levels(in, npixels, data->clipping, low, high);

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float *const restrict pixel_in = in + 4 * k;
    float *const restrict pixel_out = out + 4 * k;

    for(int c = 0; c < 3; c++)
    {
      pixel_out[c] = _map_input_level(pixel_in[c], low[c], high[c]);
    }

    pixel_out[3] = pixel_in[3];
  }
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_autoinputlevels_params_t *p = (dt_iop_autoinputlevels_params_t *)p1;
  dt_iop_autoinputlevels_data_t *d = piece->data;

  d->clipping = p->clipping;
}

void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_autoinputlevels_data_t));
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
  dt_iop_autoinputlevels_gui_data_t *g = self->gui_data;
  dt_iop_autoinputlevels_params_t *p = self->params;

  dt_bauhaus_slider_set(g->clipping, p->clipping);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_autoinputlevels_gui_data_t *g = IOP_GUI_ALLOC(autoinputlevels);

  g->clipping = dt_bauhaus_slider_from_params(self, "clipping");
  dt_bauhaus_slider_set_format(g->clipping, "%");
  dt_bauhaus_slider_set_digits(g->clipping, 3);
  gtk_widget_set_tooltip_text(g->clipping, _("histogram percentage clipped from each end"));
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
