from PIL import Image
import os
import sys

directory = sys.argv[1]
output='./output/'
for file_name in os.listdir(directory):
  print("Processing %s" % file_name)
  image = Image.open(os.path.join(directory, file_name))
  image = image.convert('RGB')
  image.save(output+file_name)


print("All done")
