import json
import logging
from typing import List, Dict, Optional
from models import KnowledgeBaseModel, PendingApprovalModel
from services.gemini_agent_service import GeminiAgentService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KnowledgeBaseService:
    def __init__(self, gemini_service: Optional[GeminiAgentService] = None):
        self.gemini_service = gemini_service or GeminiAgentService()

    def get_items(self, search_query='', limit=50):
        """Get knowledge base items"""
        return KnowledgeBaseModel.search_items(search_query, limit)
    
    def get_stats(self):
        """Get knowledge base statistics"""
        return KnowledgeBaseModel.get_stats()
    
    def get_pending_approvals(self, workflow_id=None):
        """Get pending approval items"""
        pending_items = PendingApprovalModel.get_pending_items(workflow_id)
        
        for item in pending_items:
            try:
                item['parsed_data'] = json.loads(item['item_data'])
            except:
                item['parsed_data'] = {}
        
        return pending_items
    
    def approve_items(self, workflow_id, item_ids):
        """Approve items for knowledge base"""
        pending_items_to_approve = [item for item in PendingApprovalModel.get_pending_items(workflow_id) if item['id'] in item_ids]
        
        approved_count = 0
        
        for item in pending_items_to_approve:
            try:
                item_data = json.loads(item['item_data'])
                
                KnowledgeBaseModel.add_item(
                    material_name=item_data.get('material_name'),
                    part_number=item_data.get('part_number'),
                    description=item_data.get('supplier_description'),
                    classification_label=item_data.get('qa_classification_label'),
                    confidence_level=item_data.get('qa_confidence_level'),
                    supplier_info=json.dumps({
                        'vendor_name': item_data.get('vendor_name', ''),
                        'match_source': item_data.get('match_source', ''),
                        'supplier_part_number': item_data.get('supplier_part_number', '')
                    }),
                    workflow_id=workflow_id,
                    approved_by='system',
                    metadata=json.dumps(item_data)
                )
                approved_count += 1
            except Exception as e:
                print(f"Error approving item {item['id']}: {str(e)}")
        
        PendingApprovalModel.update_approval_status(
            item_ids, 'approved', 'system', 'Approved for knowledge base'
        )
        
        return approved_count
    
    def reject_items(self, workflow_id, item_ids):
        """Reject items for knowledge base"""
        PendingApprovalModel.update_approval_status(
            item_ids, 'rejected', 'system', 'Rejected from knowledge base'
        )
        return len(item_ids)

    def search_for_matches(self, extracted_items: List[Dict]) -> List[Dict]:
        """
        Searches the knowledge base for matches to extracted items using LLM-based hybrid search.
        
        Args:
            extracted_items: A list of items extracted from a new document.
            
        Returns:
            A list of dictionaries, where each dictionary contains the original item and its best match (if found).
        """
        results = []
        # Fetch all knowledge base items to use as candidates
        kb_items = self.get_items(limit=1000)
        
        for item in extracted_items:
            # Use the LLM to find the best match based on all available metadata.
            # We pass a limited number of KB items to the LLM to manage context window size.
            # In a production system, a preliminary filter (e.g., by vendor) could be applied.
            best_match = self.gemini_service.find_best_match(
                extracted_item=item,
                kb_items=kb_items
            )

            if best_match:
                # If a match is found, add the relevant information to the result.
                results.append({
                    'original_item': item,
                    'kb_match': best_match
                })
            else:
                # No confident match found by the LLM.
                results.append({
                    'original_item': item,
                    'kb_match': None
                })
        return results
