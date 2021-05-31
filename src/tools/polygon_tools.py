import numpy as np
import bresenham


def find_first_non_zero_pixel(points, instance_image):
  points = list(points)
  coord = points[0]

  for pixel in points:
    pixel = list(pixel)
    pixel[0] = np.clip(pixel[0], 0, instance_image.shape[1]-1)
    pixel[1] = np.clip(pixel[1], 0, instance_image.shape[0]-1)
    coord = pixel

    if instance_image[pixel[1], pixel[0]] > 0:
      break

  return [int(item) for item in coord]


def find_points_from_box(box, n_points):
  assert n_points % 4 == 0, "n_points should be a multiple of four"  # simpler this way
  x0, y0, x1, y1 = box
  nbr_points = int(n_points/4)
  x_interval = (x1 - x0) / nbr_points
  y_interval = (y1 - y0) / nbr_points
  points = []
  for i in range(nbr_points):
    points.append((round(x0 + i * x_interval), y0))
  for i in range(nbr_points):
    points.append((x1, round(y0 + i * y_interval)))
  for i in range(nbr_points):
    points.append((round(x1 - i * x_interval), y1))
  for i in range(nbr_points):
    points.append((x0, round(y1 - i * y_interval)))
  return points


def mask_to_polygon(mask, bbox, nbr_vertices=16):

    points_on_box = find_points_from_box(box=bbox, n_points=nbr_vertices)
    points_on_border = []
    x0, y0, x1, y1 = bbox
    ct = int(x0 + ((x1 - x0) / 2)), int(y0 + ((y1 - y0) / 2))
    for point_on_box in points_on_box:
        line = bresenham.bresenham(int(point_on_box[0]), int(point_on_box[1]), int(ct[0]), int(ct[1]))
        points_on_border.append(find_first_non_zero_pixel(line, mask))



    return points_on_border
