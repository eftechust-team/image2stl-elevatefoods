import sys
import base64
import requests
import json

# Usage: python scripts/test_fast_generate.py path/to/image.png [output.stl]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_fast_generate.py path/to/image.png [output.stl]')
        sys.exit(1)

    image_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'out.stl'

    with open(image_path, 'rb') as f:
        data = f.read()

    b64 = base64.b64encode(data).decode('utf-8')

    payload = {
        'image': b64,
        'height': 5.0,
        'download_name': 'test_model.stl'
    }

    url = 'http://127.0.0.1:8080/fast_generate_stl'
    print('Posting to', url)
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        print('Error:', r.status_code, r.text)
        sys.exit(2)

    resp = r.json()
    if 'stl_file' not in resp:
        print('No stl_file in response:', resp)
        sys.exit(3)

    stl_b64 = resp['stl_file']
    stl_bytes = base64.b64decode(stl_b64)
    with open(out_path, 'wb') as out:
        out.write(stl_bytes)

    print('Saved STL to', out_path)
