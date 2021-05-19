import sys
import tarfile
import json

with tarfile.open(sys.argv[1], 'r') as tar:
    for member in tar.getmembers():
        print(json.dumps({'url': member.name, 'text': str(tar.extractfile(member).read())}))
