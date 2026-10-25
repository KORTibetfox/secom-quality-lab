"""Loopback-only authenticated shadow inference API."""
import math
import secrets
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Unit(BaseModel):
    model_config=ConfigDict(extra='forbid')
    unit_id:Annotated[str,Field(pattern=r'^[A-Za-z0-9_.:-]{1,64}$')]
    sensors:Annotated[list[float|None],Field(min_length=590,max_length=590)]

    @field_validator('sensors',mode='before')
    @classmethod
    def finite_numeric(cls,values):
        if not isinstance(values,list): raise ValueError('sensors must be an array')
        if any(v is not None and (type(v) not in (int,float) or not math.isfinite(v)) for v in values):
            raise ValueError('Only finite numbers or null are allowed; bool/string/NaN/Infinity rejected')
        return values


class RankRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    request_id:Annotated[str,Field(pattern=r'^[A-Za-z0-9_.:-]{1,80}$')]
    inspection_fraction:Annotated[float,Field(ge=0,le=1)]
    units:Annotated[list[Unit],Field(min_length=1,max_length=100)]

    @model_validator(mode='after')
    def unique_ids(self):
        if len({u.unit_id for u in self.units})!=len(self.units):raise ValueError('Duplicate unit_id')
        return self


class BodyLimit:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        chunks=[];total=0
        while True:
            event=await receive()
            if event['type']=='http.disconnect':return
            chunk=event.get('body',b'');total+=len(chunk)
            if total>2*1024*1024:
                return await JSONResponse({'detail':'Request body exceeds 2 MiB'},status_code=413)(scope,receive,send)
            chunks.append(chunk)
            if not event.get('more_body'):break
        delivered=False
        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered=True;return {'type':'http.request','body':b''.join(chunks),'more_body':False}
            return await receive()
        await self.app(scope,bounded_receive,send)


def create_app(runtime,token):
    if len(token)<32:raise ValueError('API token must have at least 32 characters')
    app=FastAPI(title='SECOM Shadow Operations',version='2.0',docs_url=None,redoc_url=None,openapi_url=None)
    app.add_middleware(BodyLimit)
    from fastapi.exceptions import RequestValidationError
    @app.exception_handler(RequestValidationError)
    async def validation_error(request,exc):
        # Never echo raw sensor arrays/token contents in errors.
        runtime.event('validation_error')
        return JSONResponse({'detail':'Invalid schema: check numeric values, 590 sensors, IDs and batch size'},status_code=422)
    def authorize(x_api_key:Annotated[str|None,Header()]=None):
        if not x_api_key or not secrets.compare_digest(x_api_key,token):
            runtime.event('unauthorized')
            raise HTTPException(401,'Unauthorized')
    @app.get('/health/live')
    def live():return {'status':'alive'}
    @app.get('/health/ready',dependencies=[Depends(authorize)])
    def ready():
        try:return runtime.health()
        except Exception:raise HTTPException(503,'Local state unavailable')
    @app.get('/v1/metrics',dependencies=[Depends(authorize)])
    def metrics():return runtime.metrics()
    @app.post('/v1/rank',dependencies=[Depends(authorize)])
    def rank(body:RankRequest):
        from operations import ConflictError,BusyError
        try:return runtime.score(body.model_dump())
        except ConflictError:
            runtime.event('conflict');raise HTTPException(409,'Request ID conflict')
        except BusyError:
            runtime.event('busy');raise HTTPException(503,'Capacity or state unavailable; retry with the same request ID')
        except ValueError:
            runtime.event('rejected');raise HTTPException(422,'Unusable inputs; check missing usable sensors and preserve existing inspection policy')
        except Exception:
            runtime.event('internal_error');raise HTTPException(500,'Inference unavailable; preserve existing inspection policy')
    return app
