import os
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
from typing import Optional
import uuid

# Import all services and models
from models import WorkflowModel, PendingApprovalModel
from services.translation_service import TranslationService
from services.gemini_agent_service import GeminiAgentService
from services.knowledge_base_service import KnowledgeBaseService
from services.document_parser import DocumentParser

executor = ThreadPoolExecutor(max_workers=4)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkflowService:
    """
    Main service for orchestrating the BOM processing workflow.
    It handles file uploads, document processing, data classification, and results storage.
    """
    def __init__(self):
        self.upload_dir = 'uploads'
        self.results_dir = 'results'
        self.translation_service = TranslationService()
        self.gemini_service = GeminiAgentService()
        self.kb_service = KnowledgeBaseService(gemini_service=self.gemini_service)
        self.doc_parser = DocumentParser()

    def start_workflow(self, workflow_id, wi_document, item_master=None, comparison_mode='full'):
        """
        Initializes and starts an asynchronous workflow.
        """
        try:
            workflow_dir = os.path.join(self.upload_dir, workflow_id)
            os.makedirs(workflow_dir, exist_ok=True)
            
            wi_path = os.path.join(workflow_dir, wi_document.filename)
            with open(wi_path, "wb") as buffer:
                shutil.copyfileobj(wi_document.file, buffer)
            
            item_path = None
            if item_master:
                item_path = os.path.join(workflow_dir, item_master.filename)
                with open(item_path, "wb") as buffer:
                    shutil.copyfileobj(item_master.file, buffer)
            
            WorkflowModel.create_workflow(workflow_id, comparison_mode, wi_path, item_path)
            executor.submit(self._process_workflow_async, workflow_id, wi_path, item_path, comparison_mode)
            
            return True
        except Exception as e:
            raise Exception(f"Failed to start workflow: {str(e)}")

    def _apply_classification_logic(self, item, item_master_items: list):
        """
        Applies a comprehensive set of 13 classification rules to a single item.
        This function determines the item's confidence level, classification label,
        reasoning, and action path based on a hierarchy of checks.
        
        The rules are applied in a specific order, from highest confidence to lowest.
        """
        # Define helper functions to simulate logical checks
        def _is_item_in_master(item_to_check, master_items):
            for master_item in master_items:
                pn_match = item_to_check.get('part_number') and master_item.get('part_number') and item_to_check['part_number'] == master_item['part_number']
                name_match = item_to_check.get('material_name') and master_item.get('material_name') and item_to_check['material_name'] == master_item['material_name']
                if pn_match or name_match:
                    return True
            return False

        def _is_part_number_obsolete(pn):
            # Placeholder: In a real system, this would query an obsolete parts database.
            return pn == 'OBSOLETE-PN'
            
        def _is_name_ambiguous(name):
            # Placeholder: In a real system, this would use an NLP model to score ambiguity.
            return 'Ambiguous' in name or 'Vague' in name
        
        # Merge data from item master or knowledge base if a match is found
        merged_items = self.kb_service.search_for_matches([item])
        # IMPORTANT FIX: Avoid creating a circular reference by not merging the entire `kb_match` object.
        if merged_items:
            kb_match_data = merged_items[0].get('kb_match', {})
            item['kb_match'] = kb_match_data
            item['supplier_match'] = True
        
        # Check for key data points from the extracted item
        has_pn = item.get('part_number') and item.get('part_number') != ''
        has_name = item.get('material_name') and item.get('material_name') != ''
        has_qty = item.get('qty') is not None and item.get('qty') != ''
        has_vendor = item.get('vendor_name') and item.get('vendor_name') != ''
        is_kit = item.get('kit_available', False)

        # Check against the item master
        pn_match_in_master = has_pn and item.get('part_number') in [i.get('part_number') for i in item_master_items]
        name_match_in_master = has_name and item.get('material_name') in [i.get('material_name') for i in item_master_items]

        # Initialize all flags and metadata with a default low-confidence state
        item.update({
            'pn_match': False, 'name_mismatch': False, 'pn_mismatch': False, 'obsolete_pn': False, 'vendor_name_only': False, 'kit_available': False, 'is_consumable': False,
            'qa_classification_label': '5',
            'qa_confidence_level': 'low',
            'reasoning': 'No match found',
            'action_path': '🔴 Human Intervention Required',
            'confidence_score': 0.0,
            'qa_material_name': item.get('material_name')
        })

        # --- Rule-Based Classification Logic (Ordered by Confidence) ---

        # Rule 1: Consumable/Jigs/Tools + PN + Qty
        if pn_match_in_master and has_qty:
            item.update({
                'pn_match': True,
                'is_consumable': True,
                'qa_classification_label': '1',
                'qa_confidence_level': 'high',
                'reasoning': 'Match to BOM & Item Master Data',
                'action_path': '🟢 Auto-Register',
                'confidence_score': 0.95
            })
            return item
        
        # Rule 2: Consumable/Jigs/Tools + PN + Spec + Qty
        # NOTE: This rule requires an external specification check.
        # For this implementation, we will assume a hypothetical 'spec_match' flag.
        if pn_match_in_master and has_qty and item.get('spec_match'):
            item.update({
                'pn_match': True,
                'is_consumable': True,
                'qa_classification_label': '2',
                'qa_confidence_level': 'high',
                'reasoning': 'Verify process parameters match',
                'action_path': '🟢 Auto-Register',
                'confidence_score': 0.90
            })
            return item
            
        # Rule 3: Consumable/Jigs/Tools (no qty)
        if pn_match_in_master and not has_qty:
            item.update({
                'is_consumable': True,
                'pn_match': True,
                'qa_classification_label': '3',
                'qa_confidence_level': 'medium',
                'reasoning': 'Infer qty from BOM history',
                'action_path': '🟠 Auto w/ Flag',
                'confidence_score': 0.70
            })
            return item

        # Rule 9: Vendor Name Only
        if has_vendor and not has_pn and not has_name:
            item.update({
                'vendor_name_only': True,
                'qa_classification_label': '9',
                'qa_confidence_level': 'medium',
                'reasoning': 'Map vendor to consumable in master',
                'action_path': '🟠 Auto w/ Flag',
                'confidence_score': 0.60
            })
            return item

        # Rule 11: Pre-assembled kit mentioned
        if is_kit and has_pn:
            item.update({
                'kit_available': True,
                'is_consumable': True,
                'qa_classification_label': '11',
                'qa_confidence_level': 'medium',
                'reasoning': 'Expand kit BOM',
                'action_path': '🟠 Auto w/ Flag',
                'confidence_score': 0.55
            })
            return item
            
        # Rule 12: Work Instruction mentions Consumable/Jigs/Tools only
        if not has_pn and has_name and has_qty:
            item.update({
                'is_consumable': True,
                'qa_classification_label': '12',
                'qa_confidence_level': 'medium',
                'reasoning': 'Merge WI with QC steps',
                'action_path': '🟠 Auto w/ Flag',
                'confidence_score': 0.65
            })
            return item

        # --- Low Confidence Rules ---
        
        # Rule 4: Consumable/Jigs/Tools (no Part Number)
        if not has_pn and name_match_in_master:
            item.update({
                'is_consumable': True,
                'qa_classification_label': '4',
                'qa_confidence_level': 'low',
                'reasoning': 'Check for text match in master data',
                'action_path': '🔴 Human Intervention Required',
                'confidence_score': 0.40
            })
            return item

        # Rule 6: Consumable/Jigs/Tools + Part Number mismatch
        if has_pn and has_name and not pn_match_in_master:
            item.update({
                'pn_mismatch': True,
                'is_consumable': True,
                'qa_classification_label': '6',
                'qa_confidence_level': 'low',
                'reasoning': 'Compare QC vs BOM & Master Data',
                'action_path': '🔴 Human Intervention Required',
                'confidence_score': 0.30
            })
            return item

        # Rule 7: Consumable/Jigs/Tools + Obsolete Part Number
        if _is_part_number_obsolete(item.get('part_number', '')) and has_name:
            item.update({
                'obsolete_pn': True,
                'is_consumable': True,
                'qa_classification_label': '7',
                'qa_confidence_level': 'low',
                'reasoning': 'Cross-check active/inactive status',
                'action_path': '🔴 Human Intervention Required',
                'confidence_score': 0.20
            })
            return item

        # Rule 8: Ambiguous Consumable/Jigs/Tools Name
        if _is_name_ambiguous(item.get('material_name', '')):
            item.update({
                'name_mismatch': True,
                'qa_classification_label': '8',
                'qa_confidence_level': 'low',
                'reasoning': 'NLP ambiguity score high',
                'action_path': '🔴 Human Intervention Required',
                'confidence_score': 0.35
            })
            return item

        # Rule 10: Multiple Consumable/Jigs/Tools, no mapping
        # NOTE: This is an edge case best handled by LLM extraction logic.
        # If the LLM returns an array of items, this rule applies.
        if item.get('multiple_references', False):
            item.update({
                'qa_classification_label': '10',
                'qa_confidence_level': 'low',
                'reasoning': 'Detect multiple material refs',
                'action_path': '🔴 Human Intervention Required',
                'confidence_score': 0.10
            })
            return item

        # Rule 13: Vendor + Kit Mentioned (no PN)
        if has_vendor and is_kit and not has_pn:
            item.update({
                'qa_classification_label': '13',
                'qa_confidence_level': 'low',
                'reasoning': 'Map vendor, expand kit BOM',
                'action_path': '🔴 Human Intervention Required',
                'confidence_score': 0.25
            })
            return item

        # Rule 5: No Consumable/Jigs/Tools Mentioned (Default catch-all)
        # This is the base case for all items that do not meet any other criteria.
        item.update({
            'is_consumable': False,
            'qa_classification_label': '5',
            'qa_confidence_level': 'low',
            'reasoning': 'No match found',
            'action_path': '🔴 Human Intervention Required',
            'confidence_score': 0.0
        })
        
        return item

    def _extract_and_classify_items(self, wi_content: str, item_master_items: list):
        """
        Orchestrates the extraction, enrichment, and classification of items
        from the translated document.
        """
        classified_items = []
        raw_items = self.gemini_service.extract_all_items(wi_content)
        
        for item in raw_items:
            # First, classify the raw item using the new logic
            if isinstance(item, dict):
                classified_item = self._apply_classification_logic(item, item_master_items)
                classified_items.append(classified_item)
            else:
                logging.error(f"Skipping item due to invalid format: {item}")
            
        return classified_items
    
    def _process_workflow_async(self, workflow_id, wi_path, item_path, comparison_mode):
        """
        The main asynchronous workflow execution loop.
        """
        try:
            WorkflowModel.update_workflow_status(
                workflow_id, 'processing', progress=10, 
                stage='extracting', message='Extracting data from documents'
            )
            
            wi_content = self.doc_parser.extract_text(wi_path)
            item_master_items = self.doc_parser.parse_item_master(item_path, self.gemini_service) if item_path else []
            
            logging.info(f"Workflow {workflow_id}: Document content extracted.")
            logging.info(f"Extracted WI Content:\n{wi_content}")
            logging.info(f"Extracted Item Master Content:\n{item_master_items}")

            WorkflowModel.update_workflow_status(
                workflow_id, 'processing', progress=30, 
                stage='translating', message='Translating document to English'
            )
            
            translated_wi_content = self.translation_service.translate_to_english(wi_content)
            logging.info(f"Workflow {workflow_id}: Document translated. Logged to results.")
            logging.info(f"Translated Content:\n{translated_wi_content}")

            WorkflowModel.update_workflow_status(
                workflow_id, 'processing', progress=50, 
                stage='classifying', message='Classifying and matching items with Gemini'
            )

            # Pass the extracted and standardized item master data to the classification logic
            extracted_items = self._extract_and_classify_items(translated_wi_content, item_master_items)
            
            logging.info(f"Workflow {workflow_id}: Gemini agent completed. Extracted {len(extracted_items)} items.")
            logging.info(f"Extracted Items:\n{extracted_items}")
            
            summary = self._generate_summary(extracted_items, comparison_mode)
            self._save_workflow_results(workflow_id, extracted_items, summary)
            self._create_pending_approvals(workflow_id, extracted_items)
            
            WorkflowModel.update_workflow_status(
                workflow_id, 'completed', progress=100, 
                stage='completed', message='Processing completed successfully'
            )
            
        except Exception as e:
            WorkflowModel.update_workflow_status(
                workflow_id, 'error', message=f'Processing failed: {str(e)}'
            )
            logging.error(f"Workflow {workflow_id} failed with error: {e}")

    def _extract_text_from_document(self, file_path):
        return self.doc_parser.extract_text(file_path)

    def _extract_text_from_excel(self, file_path):
        return self.doc_parser.extract_text(file_path)

    def _generate_summary(self, items, comparison_mode):
        if not isinstance(items, list):
            return {
                'total_materials': 0,
                'successful_matches': 0,
                'knowledge_base_matches': 0,
                'comparison_mode': comparison_mode
            }
        
        total_materials = len(items)
        successful_matches = sum(1 for item in items if isinstance(item, dict) and item.get('qa_confidence_level') in ['high', 'medium'])
        knowledge_base_matches = sum(1 for item in items if isinstance(item, dict) and 'knowledge_base' in item.get('reasoning', '').lower())
        
        return {
            'total_materials': total_materials,
            'successful_matches': successful_matches,
            'knowledge_base_matches': knowledge_base_matches,
                'comparison_mode': comparison_mode
        }

    def _save_workflow_results(self, workflow_id, results, summary):
        results_file = os.path.join(self.results_dir, f'{workflow_id}.json')
        with open(results_file, 'w') as f:
            json.dump({'matches': results, 'summary': summary}, f, indent=2)
        
        from models import get_db_connection
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO workflow_results (workflow_id, results_data, summary_data)
            VALUES (?, ?, ?)
        ''', (workflow_id, json.dumps({'matches': results}), json.dumps(summary)))
        conn.commit()
        conn.close()
    
    def _create_pending_approvals(self, workflow_id, matches):
        for match in matches:
            if isinstance(match, dict) and match.get('qa_confidence_level') in ['high', 'medium', 'low']:
                PendingApprovalModel.add_pending_item(workflow_id, json.dumps(match))
    
    def get_workflow_status(self, workflow_id):
        workflow = WorkflowModel.get_workflow(workflow_id)
        if not workflow:
            raise ValueError("Workflow not found")
        return workflow
    
    def get_workflow_results(self, workflow_id):
        results_file = os.path.join(self.results_dir, f'{workflow_id}.json')
        if not os.path.exists(results_file):
            raise ValueError("Results not found")
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def get_all_workflows(self):
        workflows = WorkflowModel.get_all_workflows()
        
        for workflow in workflows:
            results_file = os.path.join(self.results_dir, f"{workflow['id']}.json")
            workflow['has_results'] = os.path.exists(results_file)
        
        return workflows
