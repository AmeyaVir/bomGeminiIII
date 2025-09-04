from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import uuid
from typing import Optional

from services.workflow_service import WorkflowService
from services.knowledge_base_service import KnowledgeBaseService

app = FastAPI(
    title="BOM Platform API",
    description="Backend API for the autonomous BOM processing platform with Gemini integration.",
    version="4.0.0",
)

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
workflow_service = WorkflowService()
kb_service = KnowledgeBaseService()

@app.on_event("startup")
async def startup_event():
    """Initializes the database and creates directories on startup."""
    try:
        from models import init_db
        init_db()
        os.makedirs(workflow_service.upload_dir, exist_ok=True)
        os.makedirs(workflow_service.results_dir, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server startup failed: {e}")

@app.get("/api/workflows")
async def get_workflows():
    """Get all workflows from the database."""
    try:
        workflows = workflow_service.get_all_workflows()
        return JSONResponse(content={'success': True, 'workflows': workflows})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge-base")
async def get_knowledge_base(search: Optional[str] = "", limit: int = 50):
    """Get knowledge base items with statistics, with optional search."""
    try:
        items = kb_service.get_items(search, limit)
        stats = kb_service.get_stats()
        return JSONResponse(content={'success': True, 'items': items, 'stats': stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge-base/pending")
async def get_pending_approvals():
    """Get pending items for approval."""
    try:
        pending_items = kb_service.get_pending_approvals()
        return JSONResponse(content={'success': True, 'pending_items': pending_items})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge-base/approve")
async def approve_knowledge_base_item(workflow_id: str, item_ids: list[int]):
    """Approve an item for the knowledge base."""
    try:
        result = kb_service.approve_items(workflow_id, item_ids)
        return JSONResponse(content={'success': True, 'approved_count': result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge-base/reject")
async def reject_knowledge_base_item(workflow_id: str, item_ids: list[int]):
    """Reject an item from the knowledge base."""
    try:
        result = kb_service.reject_items(workflow_id, item_ids)
        return JSONResponse(content={'success': True, 'rejected_count': result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/autonomous/upload")
async def upload_documents(
    wi_document: UploadFile = File(..., description="The Japanese WI/QC document to process."),
    item_master: Optional[UploadFile] = File(None, description="Optional Item Master for full comparison mode."),
    comparison_mode: str = Form(..., description="'full' or 'kb_only'")
):
    """Enhanced upload endpoint with optional Item Master and Gemini processing."""
    try:
        if not wi_document:
            raise HTTPException(status_code=400, detail="WI document is required")

        if comparison_mode == 'full' and not item_master:
            raise HTTPException(status_code=400, detail="Item Master is required for full comparison mode")

        workflow_id = str(uuid.uuid4())

        # Start processing asynchronously
        workflow_service.start_workflow(
            workflow_id=workflow_id,
            wi_document=wi_document,
            item_master=item_master,
            comparison_mode=comparison_mode
        )

        return JSONResponse(content={
            'success': True,
            'workflow_id': workflow_id,
            'message': 'Processing started successfully'
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")

@app.get("/api/autonomous/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow status."""
    try:
        status = workflow_service.get_workflow_status(workflow_id)
        return JSONResponse(content=status)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {str(e)}")

@app.get("/api/autonomous/workflow/{workflow_id}/results")
async def get_workflow_results(workflow_id: str):
    """Get workflow results."""
    try:
        results = workflow_service.get_workflow_results(workflow_id)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Results not found: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
     
