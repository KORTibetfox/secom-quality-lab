"""Download the attributed UCI source and verify the recorded SHA-256 values."""
import argparse,hashlib,io,json,os,tempfile,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URL='https://archive.ics.uci.edu/static/public/179/secom.zip'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,default=ROOT/'data/raw');args=parser.parse_args()
    expected=json.loads((ROOT/'results/protocol.json').read_text(encoding='utf-8'))['source_sha256']
    target=args.destination;target.mkdir(parents=True,exist_ok=True)
    def valid(path,digest):return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    if all(valid(target/name,digest) for name,digest in expected.items()):
        print('Original UCI files already match all three recorded hashes.');return
    with urllib.request.urlopen(URL,timeout=60) as response:data=response.read(10*1024*1024+1)
    if len(data)>10*1024*1024:raise ValueError('Unexpected source archive size')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        verified={}
        for name,digest in expected.items():
            matches=[i for i in archive.infolist() if Path(i.filename).name==name]
            if len(matches)!=1 or matches[0].file_size>10*1024*1024:raise ValueError('Unexpected archive contents')
            content=archive.read(matches[0])
            if hashlib.sha256(content).hexdigest()!=digest:raise ValueError(f'Hash mismatch: {name}; existing files retained')
            verified[name]=content
        for name,content in verified.items():
            fd,tmp=tempfile.mkstemp(prefix='.download-',dir=target)
            with os.fdopen(fd,'wb') as stream:stream.write(content)
            os.replace(tmp,target/name)
    print('Downloaded and verified three original UCI files.')
if __name__=='__main__':main()
