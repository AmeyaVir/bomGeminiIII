import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Upload, File, Database, ToggleLeft, ToggleRight } from 'lucide-react';
import { useTranslation } from '../contexts/TranslationContext';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import toast from 'react-hot-toast';

function UploadPage() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const [wiDocument, setWiDocument] = useState(null);
  const [itemMaster, setItemMaster] = useState(null);
  const [comparisonMode, setComparisonMode] = useState('full');
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (event, type) => {
    const file = event.target.files[0];
    if (type === 'wi') {
      setWiDocument(file);
    } else {
      setItemMaster(file);
    }
  };

  const toggleComparisonMode = () => {
    setComparisonMode(comparisonMode === 'full' ? 'kb_only' : 'full');
    if (comparisonMode === 'full') {
      setItemMaster(null);
    }
  };

  const handleUpload = async () => {
    if (!wiDocument) {
      toast.error('Please select a WI document');
      return;
    }
    if (comparisonMode === 'full' && !itemMaster) {
      toast.error('Please select an Item Master for full comparison mode');
      return;
    }
    try {
      setUploading(true);
      const formData = new FormData();
      formData.append('wi_document', wiDocument);
      formData.append('comparison_mode', comparisonMode);
      
      if (itemMaster) {
        formData.append('item_master', itemMaster);
      }
      const response = await fetch('/api/autonomous/upload', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }
      const data = await response.json();
      if (data.success) {
        toast.success('Upload successful!');
        navigate(`/processing/${data.workflow_id}`);
      } else {
        throw new Error(data.message || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Upload failed: ' + error.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto p-6">
        <div className="mb-8">
          <button onClick={() => navigate('/dashboard')} className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4">
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back to Dashboard
          </button>
          <h1 className="text-3xl font-bold text-gray-900">{t('navigation.upload')}</h1>
          <p className="text-gray-600 mt-1">
            Upload your Japanese WI/QC document and optionally an Item Master for autonomous processing
          </p>
        </div>
        <Card className="p-6 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-medium text-gray-900">Comparison Mode</h3>
              <p className="text-sm text-gray-500 mt-1">
                Choose how to compare your WI document
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <span className={`text-sm font-medium ${
                comparisonMode === 'full' ? 'text-primary-600' : 'text-gray-500'
              }`}>
                Full Comparison
              </span>
              <button onClick={toggleComparisonMode} className="focus:outline-none">
                {comparisonMode === 'kb_only' ? (
                  <ToggleRight className="h-8 w-8 text-primary-600" />
                ) : (
                  <ToggleLeft className="h-8 w-8 text-gray-400" />
                )}
              </button>
              <span className={`text-sm font-medium ${
                comparisonMode === 'kb_only' ? 'text-primary-600' : 'text-gray-500'
              }`}>
                Knowledge Base Only
              </span>
            </div>
          </div>
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            {comparisonMode === 'full' ? (
              <div className="flex items-start space-x-3">
                <File className="h-5 w-5 text-blue-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-gray-900">Full Comparison Mode</p>
                  <p className="text-sm text-gray-600">
                    Compare WI document against both Item Master and Knowledge Base for comprehensive matching
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex items-start space-x-3">
                <Database className="h-5 w-5 text-purple-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-gray-900">Knowledge Base Only Mode</p>
                  <p className="text-sm text-gray-600">
                    Compare WI document against historical knowledge base only - no Item Master required
                  </p>
                </div>
              </div>
            )}
          </div>
        </Card>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <Card className="p-6">
            <div className="text-center">
              <File className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                WI/QC Document *
              </h3>
              <p className="text-sm text-gray-500 mb-4">
                Supports PDF, DOCX, DOC, TXT formats
              </p>
              <input
                type="file"
                accept=".pdf,.docx,.doc,.txt"
                onChange={(e) => handleFileChange(e, 'wi')}
                className="hidden"
                id="wi-upload"
              />
              <label
                htmlFor="wi-upload"
                className="cursor-pointer inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <Upload className="h-4 w-4 mr-2" />
                Choose File
              </label>
              {wiDocument && (
                <div className="mt-4">
                  <p className="text-sm text-green-600 font-medium">Selected:</p>
                  <p className="text-sm text-gray-700">{wiDocument.name}</p>
                  <p className="text-xs text-gray-500">
                    {(wiDocument.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              )}
            </div>
          </Card>
          <Card className={`p-6 ${comparisonMode === 'kb_only' ? 'opacity-50' : ''}`}>
            <div className="text-center">
              <File className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Item Master {comparisonMode === 'full' ? '*' : '(Optional)'}
              </h3>
              <p className="text-sm text-gray-500 mb-4">
                {comparisonMode === 'kb_only'
                  ? 'Not required in Knowledge Base Only mode'
                  : 'Supports Excel (XLSX, XLS) and CSV formats'
                }
              </p>
              <input
                type="file"
                accept=".xlsx,.xls,.csv"
                onChange={(e) => handleFileChange(e, 'item')}
                className="hidden"
                id="item-upload"
                disabled={comparisonMode === 'kb_only'}
              />
              <label
                htmlFor="item-upload"
                className={`cursor-pointer inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 ${
                  comparisonMode === 'kb_only' ? 'cursor-not-allowed opacity-50' : ''
                }`}
              >
                <Upload className="h-4 w-4 mr-2" />
                Choose File
              </label>
              {itemMaster && comparisonMode === 'full' && (
                <div className="mt-4">
                  <p className="text-sm text-green-600 font-medium">Selected:</p>
                  <p className="text-sm text-gray-700">{itemMaster.name}</p>
                  <p className="text-xs text-gray-500">
                    {(itemMaster.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              )}
              {comparisonMode === 'kb_only' && (
                <div className="mt-4 text-xs text-gray-500">
                  Knowledge Base Only mode selected
                </div>
              )}
            </div>
          </Card>
        </div>
        <div className="text-center">
          <Button
            onClick={handleUpload}
            disabled={!wiDocument || (comparisonMode === 'full' && !itemMaster) || uploading}
            className="btn-primary px-8 py-3"
            size="lg"
          >
            {uploading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Processing...
              </>
            ) : (
              <>
                <Upload className="h-5 w-5 mr-2" />
                Start Processing
              </>
            )}
          </Button>
          <div className="mt-6 max-w-2xl mx-auto">
            <p className="text-sm text-gray-500">
              Our autonomous agents will process your documents through translation, extraction with
              WI/QC Item classification, and intelligent comparison stages.
              {comparisonMode === 'kb_only'
                ? ' Items will be compared against our knowledge base only.'
                : ' Items will be compared against both your Item Master and our knowledge base.'
              }
              You can monitor progress in real-time.
            </p>
            {comparisonMode === 'kb_only' && (
              <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
                <div className="flex items-center">
                  <Database className="h-4 w-4 text-blue-500 mr-2" />
                  <p className="text-sm text-blue-700 font-medium">
                    Knowledge Base Only Mode
                  </p>
                </div>
                <p className="text-sm text-blue-600 mt-1">
                  After processing, you'll be able to approve which items should be added to the knowledge base.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default UploadPage;
