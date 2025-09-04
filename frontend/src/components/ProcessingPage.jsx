import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, CheckCircle } from 'lucide-react';
import { useTranslation } from '../contexts/TranslationContext';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import Card from './ui/Card';

function ProcessingPage() {
  const { workflowId } = useParams();
  const navigate = useNavigate();
  const { t } = useTranslation();
  const [workflow, setWorkflow] = useState(null);
  const [loading, setLoading] = useState(true);

  const checkWorkflowStatus = async () => {
    try {
      const response = await fetch(`/api/autonomous/workflow/${workflowId}/status`);
      if (response.ok) {
        const data = await response.json();
        setWorkflow(data);
        
        if (data.status === 'completed') {
          setTimeout(() => {
            navigate(`/results/${workflowId}`);
          }, 1000);
        } else if (data.status === 'error') {
          setLoading(false);
        }
      }
    } catch (error) {
      console.error('Failed to check workflow status:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (workflowId) {
      const interval = setInterval(() => {
        checkWorkflowStatus();
      }, 2000);
      
      checkWorkflowStatus();
      return () => clearInterval(interval);
    }
  }, [workflowId, navigate, checkWorkflowStatus]);

  if (loading && !workflow) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="lg" className="mx-auto mb-4" />
          <p className="text-gray-600">{t('common.loading')}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto p-6">
        <div className="mb-8">
          <button onClick={() => navigate('/dashboard')} className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4">
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back to Dashboard
          </button>
          <h1 className="text-3xl font-bold text-gray-900">{t('navigation.processing')}</h1>
          <p className="text-gray-600 mt-1">Workflow ID: {workflowId}</p>
        </div>
        {workflow && (
          <Card className="p-8">
            <div className="text-center">
              {workflow.status === 'completed' ? (
                <CheckCircle className="mx-auto h-16 w-16 text-green-500 mb-4" />
              ) : workflow.status === 'error' ? (
                <div className="mx-auto h-16 w-16 text-red-500 mb-4">❌</div>
              ) : (
                <LoadingSpinner size="lg" className="mx-auto mb-4" />
              )}
              <h2 className="text-xl font-semibold text-gray-900 mb-2">
                {workflow.status === 'completed'
                  ? 'Processing Complete!'
                  : workflow.status === 'error'
                  ? 'Processing Failed'
                  : 'Processing in Progress...'
                }
              </h2>
              <p className="text-gray-600 mb-6">
                {workflow.message || 'Processing your documents...'}
              </p>
              <div className="bg-gray-200 rounded-full h-2 mb-4">
                <div
                  className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${workflow.progress || 0}%` }}
                ></div>
              </div>
              <p className="text-sm text-gray-500">
                Current Stage: {workflow.current_stage || 'initializing'}
              </p>
              {workflow.progress && (
                <p className="text-sm text-gray-500 mt-1">
                  {Math.round(workflow.progress)}% Complete
                </p>
              )}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

export default ProcessingPage;
