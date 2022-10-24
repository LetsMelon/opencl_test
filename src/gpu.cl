// Unused... here for comparison purposes:
__constant sampler_t sampler_const =
  CLK_NORMALIZED_COORDS_FALSE |
  CLK_ADDRESS_NONE |
  CLK_FILTER_NEAREST;

float map_value(float value, float low1, float high1, float low2, float high2)
{
  float back_value = low2 + (value - low1) * (high2 - low2) / (high1 - low1);
  return back_value;
}

__kernel void my_fct(
    sampler_t sampler_host,
    int run,
    int run_max,
    read_only image2d_t src_image,
    write_only image2d_t dst_image) 
  {
  int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));
  int2 image_dims =  get_image_dim(src_image);

  float r = map_value(pixelcoord.x, 0.0, image_dims.x - 1, 0.0, 1.0);
  float g = map_value(pixelcoord.y, 0.0, image_dims.y - 1, 0.0, 1.0);
  float b = map_value((float)run, 0.0, run_max, 0.0, 1.0);
  float a = 1.0;

  float4 pixel = (float4)(r, g, b, a);

  write_imagef(dst_image, pixelcoord, pixel);
}
