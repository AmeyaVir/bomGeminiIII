import sqlite3
import os
import json
from datetime import datetime

DB_PATH = 'bom_platform.db'

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with all tables"""
    conn = get_db_connection()
    
    # Workflows table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'pending',
            comparison_mode TEXT NOT NULL DEFAULT 'full',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            progress INTEGER DEFAULT 0,
            current_stage TEXT,
            message TEXT,
            wi_document_path TEXT,
            item_master_path TEXT
        )
    ''')
    
    # Knowledge base table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_name TEXT NOT NULL,
            part_number TEXT,
            description TEXT,
            classification_label INTEGER,
            confidence_level TEXT,
            supplier_info TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            workflow_id TEXT,
            approved_by TEXT,
            approved_at TIMESTAMP,
            metadata TEXT
        )
    ''')
    
    # Pending approvals table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS pending_approvals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_id TEXT NOT NULL,
            item_data TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_by TEXT,
            reviewed_at TIMESTAMP,
            review_notes TEXT
        )
    ''')
    
    # Workflow results table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS workflow_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_id TEXT NOT NULL,
            results_data TEXT NOT NULL,
            summary_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

class WorkflowModel:
    @staticmethod
    def create_workflow(workflow_id, comparison_mode='full', wi_path=None, item_path=None):
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO workflows (id, comparison_mode, wi_document_path, item_master_path)
            VALUES (?, ?, ?, ?)
        ''', (workflow_id, comparison_mode, wi_path, item_path))
        conn.commit()
        conn.close()
    
    @staticmethod
    def update_workflow_status(workflow_id, status, progress=None, stage=None, message=None):
        conn = get_db_connection()
        updates = ['status = ?', 'updated_at = CURRENT_TIMESTAMP']
        values = [status]
        
        if progress is not None:
            updates.append('progress = ?')
            values.append(progress)
        if stage:
            updates.append('current_stage = ?')
            values.append(stage)
        if message:
            updates.append('message = ?')
            values.append(message)
        
        values.append(workflow_id)
        
        conn.execute(f'''
            UPDATE workflows SET {', '.join(updates)}
            WHERE id = ?
        ''', values)
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_workflow(workflow_id):
        conn = get_db_connection()
        workflow = conn.execute('''
            SELECT * FROM workflows WHERE id = ?
        ''', (workflow_id,)).fetchone()
        conn.close()
        return dict(workflow) if workflow else None
    
    @staticmethod
    def get_all_workflows(limit=50):
        conn = get_db_connection()
        workflows = conn.execute('''
            SELECT * FROM workflows 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,)).fetchall()
        conn.close()
        return [dict(w) for w in workflows]
    
class KnowledgeBaseModel:
    @staticmethod
    def add_item(material_name, part_number=None, description=None, 
                classification_label=None, confidence_level=None, 
                supplier_info=None, workflow_id=None, approved_by=None, metadata=None):
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO knowledge_base 
            (material_name, part_number, description, classification_label, 
             confidence_level, supplier_info, workflow_id, approved_by, 
             approved_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        ''', (material_name, part_number, description, classification_label,
              confidence_level, supplier_info, workflow_id, approved_by, metadata))
        conn.commit()
        conn.close()
    
    @staticmethod
    def search_items(query='', limit=50):
        conn = get_db_connection()
        if query:
            items = conn.execute('''
                SELECT * FROM knowledge_base 
                WHERE material_name LIKE ? OR part_number LIKE ? OR description LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit)).fetchall()
        else:
            items = conn.execute('''
                SELECT * FROM knowledge_base 
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,)).fetchall()
        conn.close()
        return [dict(item) for item in items]
    
    @staticmethod
    def get_stats():
        conn = get_db_connection()
        total_items = conn.execute('SELECT COUNT(*) as count FROM knowledge_base').fetchone()['count']
        total_workflows = conn.execute('''
            SELECT COUNT(DISTINCT workflow_id) as count FROM knowledge_base 
            WHERE workflow_id IS NOT NULL
        ''').fetchone()['count']
        total_matches = total_items
        high_confidence_items = conn.execute('''
            SELECT COUNT(*) as count FROM knowledge_base 
            WHERE confidence_level = 'high'
        ''').fetchone()['count']
        
        match_rate = (high_confidence_items / total_items * 100) if total_items > 0 else 0
        conn.close()
        
        return {
            'total_items': total_items,
            'total_workflows': total_workflows,
            'total_matches': total_matches,
            'match_rate': round(match_rate, 1)
        }

class PendingApprovalModel:
    @staticmethod
    def add_pending_item(workflow_id, item_data):
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO pending_approvals (workflow_id, item_data)
            VALUES (?, ?)
        ''', (workflow_id, item_data))
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_pending_items(workflow_id=None):
        conn = get_db_connection()
        if workflow_id:
            items = conn.execute('''
                SELECT * FROM pending_approvals 
                WHERE workflow_id = ? AND status = 'pending'
                ORDER BY created_at DESC
            ''', (workflow_id,)).fetchall()
        else:
            items = conn.execute('''
                SELECT * FROM pending_approvals 
                WHERE status = 'pending'
                ORDER BY created_at DESC
            ''').fetchall()
        conn.close()
        return [dict(item) for item in items]
    
    @staticmethod
    def update_approval_status(item_ids, status, reviewer=None, notes=None):
        conn = get_db_connection()
        placeholders = ','.join(['?' for _ in item_ids])
        conn.execute(f'''
            UPDATE pending_approvals 
            SET status = ?, reviewed_by = ?, reviewed_at = CURRENT_TIMESTAMP, review_notes = ?
            WHERE id IN ({placeholders})
        ''', [status, reviewer, notes] + item_ids)
        conn.commit()
        conn.close()
     
