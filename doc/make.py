import subprocess
import os

from distutils.dir_util import copy_tree


# Start building documentation.
builder = subprocess.Popen(['sphinx-build', '-M', 'html', '.', '_build'])
builder.communicate()
builder.wait()

# Copy output to target repository.
target_repository = '../../trackintel-docs'
if not (os.path.exists(target_repository) and os.path.isdir(target_repository)):
    raise NotADirectoryError("Target repository does not exist. Please clone to the right location first.")
else:
    copy_tree('_build/html', target_repository)
