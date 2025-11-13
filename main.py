# main.py — PropPulse Backend (FastAPI)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Any

import prop_ev as pe  # your existing model file

app = FastAPI(
    title="PropPulse Backend API",
    version="1.0.0",
    description="FastAPI wrapper around PropPulse prop_ev model."
)

# Load settings once at startup
SETTINGS = pe.load_settings() if hasattr(pe, "load_settings") else None


# ============
# Request / Response models
# ============

class SinglePropRequest(BaseModel):
    player: str
    stat: str          # "PTS", "REB", "AST", "PRA", etc.
    line: float
    odds: int = -110   # default
    debug: bool = False


class SinglePropResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    result: Optional[dict] = None


class BatchPropRequest(BaseModel):
    props: List[SinglePropRequest]


class BatchPropResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    results: Optional[List[dict]] = None


# ============
# Health check
# ============

@app.get("/")
def root():
    return {"status": "ok", "service": "PropPulse Backend", "version": "1.0.0"}


# ============
# Single prop endpoint
# ============

@app.post("/analyze", response_model=SinglePropResponse)
def analyze_single(request: SinglePropRequest):
    try:
        settings = SETTINGS or (pe.load_settings() if hasattr(pe, "load_settings") else None)

        # Adjust this call if your signature is different
        result = pe.analyze_single_prop(
            player=request.player,
            stat=request.stat,
            line=request.line,
            odds=request.odds,
            settings=settings,
            debug_mode=request.debug,
        )

        # Make sure it's JSON-serializable
        if hasattr(result, "to_dict"):
            result = result.to_dict()

        return SinglePropResponse(success=True, result=result)

    except Exception as e:
        return SinglePropResponse(success=False, error=str(e))


# ============
# Batch endpoint
# ============

@app.post("/analyze-batch", response_model=BatchPropResponse)
def analyze_batch(request: BatchPropRequest):
    try:
        settings = SETTINGS or (pe.load_settings() if hasattr(pe, "load_settings") else None)
        results: List[dict] = []

        for p in request.props:
            try:
                r = pe.analyze_single_prop(
                    player=p.player,
                    stat=p.stat,
                    line=p.line,
                    odds=p.odds,
                    settings=settings,
                    debug_mode=p.debug,
                )
                if hasattr(r, "to_dict"):
                    r = r.to_dict()
                results.append(r)
            except Exception as inner_e:
                results.append({"error": str(inner_e), "player": p.player})

        return BatchPropResponse(success=True, results=results)

    except Exception as e:
        return BatchPropResponse(success=False, error=str(e))
