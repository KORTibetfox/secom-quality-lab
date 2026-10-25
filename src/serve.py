"""Never opens a LAN/public listener. No auto model promotion or reload."""
import argparse
import os
import secrets
from pathlib import Path
import uvicorn
from quality import ROOT
from operations import Runtime
from api import create_app


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    state=ROOT/'state';state.mkdir(exist_ok=True)
    token=os.environ.get('SECOM_API_TOKEN')
    if not token:
        path=state/'.api_token'
        if not path.exists():
            with path.open('x',encoding='utf-8') as f:f.write(secrets.token_urlsafe(32))
        token=path.read_text(encoding='utf-8').strip()
    app=create_app(Runtime(),token)
    print('Shadow-only API: http://127.0.0.1:'+str(args.port)+'; token stored locally, never printed.',flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port,workers=1,access_log=False,limit_concurrency=8,timeout_keep_alive=5)


if __name__=='__main__':main()
