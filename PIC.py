from PIL import Image
import os
import sys

directory = sys.argv[1]

for file_name in os.listdir(directory):
  print("Processing %s" % file_name)
  image = Image.open(os.path.join(directory, file_name))

  new_dimensions = (200, 250)
  output = image.resize(new_dimensions, Image.ANTIALIAS)

  output_file_name = os.path.join(directory,file_name)
  output.save(output_file_name, "JPEG", quality = 95)

print("All done")
