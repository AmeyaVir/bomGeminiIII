import os
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd
import logging
from typing import Optional
import uuid

from models import WorkflowModel, PendingApprovalModel
from services.translation_service import TranslationService
from services.gemini_agent_service import GeminiAgentService
from services.knowledge_base_service import KnowledgeBaseService
from services.document_parser import DocumentParser

executor = ThreadPoolExecutor(max_workers=4)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkflowService:
    def __init__(self):
        self.upload_dir = 'uploads'
        self.results_dir = 'results'
        self.translation_service = TranslationService()
        self.gemini_service = GeminiAgentService()
        self.kb_service = KnowledgeBaseService()
        self.doc_parser = DocumentParser()

    def start_workflow(self, workflow_id, wi_document, item_master=None, comparison_mode='full'):
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

    def _apply_classification_logic(self, item, item_master_items_pn_set, item_master_items_name_set):
        pn_match = item.get('part_number') and item.get('part_number') in item_master_items_pn_set
        name_match = item.get('material_name') and item.get('material_name') in item_master_items_name_set
        qty_present = item.get('qty') is not None and item.get('qty') != ''
        
        item['qa_classification_label'] = '5'
        item['qa_confidence_level'] = 'low'
        item['reasoning'] = 'No match found'
        item['action_path'] = '🔴 Human Intervention Required'
        
        # Rule 1: Consumable/Jigs/Tools + PN + Qty
        if pn_match and qty_present:
            item.update({
                'qa_classification_label': '1',
                'qa_confidence_level': 'high',
                'reasoning': 'Match to BOM & Item Master Data',
                'action_path': '🟢 Auto-Register'
            })
        # Rule 4: Consumable/Jigs/Tools (no Part Number)
        elif not pn_match and name_match:
            item.update({
                'qa_classification_label': '4',
                'qa_confidence_level': 'low',
                'reasoning': 'Check for text match in master data',
                'action_path': '🔴 Human Intervention Required'
            })
        
        return item

    def _extract_and_classify_items(self, wi_content: str, item_master_items: list):
        classified_items = []
        raw_items = self.gemini_service.extract_all_items(wi_content)
        
        item_master_items_pn_set = {item['part_number'] for item in item_master_items if item.get('part_number')}
        item_master_items_name_set = {item['material_name'] for item in item_master_items if item.get('material_name')}
        
        for item in raw_items:
            classified_item = self._apply_classification_logic(item, item_master_items_pn_set, item_master_items_name_set)
            classified_items.append(classified_item)
            
        return classified_items
    
    def _process_workflow_async(self, workflow_id, wi_path, item_path, comparison_mode):
        try:
            WorkflowModel.update_workflow_status(
                workflow_id, 'processing', progress=10, 
                stage='extracting', message='Extracting data from documents'
            )
            
            wi_content = self.doc_parser.extract_text(wi_path)
            item_master_items = self.doc_parser.parse_item_master(item_path) if item_path else []
            
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

            item_master_items = self.doc_parser.parse_item_master(item_path) if item_path else []
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
            if isinstance(match, dict) and match.get('qa_confidence_level') in ['low','high', 'medium']:
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
