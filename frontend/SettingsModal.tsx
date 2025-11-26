import React, { useState, useEffect } from 'react';
import { X, Save, Key, Server, AlertCircle, CheckCircle2 } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [apiKey, setApiKey] = useState('');
  const [forwardEndpoint, setForwardEndpoint] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [hasApiKey, setHasApiKey] = useState(false);

  const isElectron = typeof window !== 'undefined' && window.electronAPI !== undefined;

  useEffect(() => {
    if (isOpen && isElectron) {
      loadSettings();
    }
  }, [isOpen, isElectron]);

  const loadSettings = async () => {
    if (!window.electronAPI) return;

    setIsLoading(true);
    try {
      const [storedApiKey, storedForwardEndpoint] = await Promise.all([
        window.electronAPI.getApiKey(),
        window.electronAPI.getForwardEndpoint()
      ]);

      // Don't show the actual API key, just indicate if one exists
      setHasApiKey(!!storedApiKey);
      setApiKey(''); // Clear the input
      setForwardEndpoint(storedForwardEndpoint || 'https://api.openai.com/v1');
    } catch (error) {
      console.error('Error loading settings:', error);
      setMessage({ type: 'error', text: 'Failed to load settings' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    if (!window.electronAPI) return;

    setIsSaving(true);
    setMessage(null);

    try {
      // Validate forward endpoint
      if (forwardEndpoint && forwardEndpoint.trim()) {
        try {
          new URL(forwardEndpoint.trim());
        } catch {
          setMessage({ type: 'error', text: 'Invalid forward endpoint format' });
          setIsSaving(false);
          return;
        }
      }

      // Save API key if provided
      if (apiKey.trim()) {
        const result = await window.electronAPI.setApiKey(apiKey.trim());
        if (!result.success) {
          setMessage({ type: 'error', text: result.error || 'Failed to save API key' });
          setIsSaving(false);
          return;
        }
        setHasApiKey(true);
        setApiKey(''); // Clear after saving
      }

      // Save forward endpoint
      const urlResult = await window.electronAPI.setForwardEndpoint(forwardEndpoint.trim());
      if (!urlResult.success) {
        setMessage({ type: 'error', text: urlResult.error || 'Failed to save forward endpoint' });
        setIsSaving(false);
        return;
      }

      setMessage({ type: 'success', text: 'Settings saved successfully!' });
      setTimeout(() => {
        onClose();
      }, 1000);
    } catch (error) {
      console.error('Error saving settings:', error);
      setMessage({ type: 'error', text: 'Failed to save settings' });
    } finally {
      setIsSaving(false);
    }
  };

  const handleClearApiKey = async () => {
    if (!window.electronAPI) return;

    setIsSaving(true);
    setMessage(null);

    try {
      const result = await window.electronAPI.setApiKey('');
      if (result.success) {
        setHasApiKey(false);
        setApiKey('');
        setMessage({ type: 'success', text: 'API key cleared' });
      } else {
        setMessage({ type: 'error', text: result.error || 'Failed to clear API key' });
      }
    } catch (error) {
      console.error('Error clearing API key:', error);
      setMessage({ type: 'error', text: 'Failed to clear API key' });
    } finally {
      setIsSaving(false);
    }
  };

  if (!isOpen) return null;

  if (!isElectron) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-slate-800">Settings</h2>
            <button
              onClick={onClose}
              className="text-slate-500 hover:text-slate-700 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
          <p className="text-slate-600">Settings are only available in Electron mode.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl p-6 max-w-lg w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-slate-800">Settings</h2>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-700 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <div className="space-y-6">
            {/* OpenAI API Key */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
                <Key className="w-4 h-4" />
                OpenAI API Key
              </label>
              <div className="space-y-2">
                {hasApiKey && !apiKey && (
                  <div className="flex items-center gap-2 text-sm text-green-600 bg-green-50 p-2 rounded">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>API key is configured</span>
                  </div>
                )}
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={hasApiKey ? "Enter new API key to update" : "Enter your OpenAI API key"}
                  className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-blue-500 focus:outline-none font-mono text-sm"
                />
                {hasApiKey && (
                  <button
                    onClick={handleClearApiKey}
                    className="text-sm text-red-600 hover:text-red-700 transition-colors"
                  >
                    Clear API key
                  </button>
                )}
                <p className="text-xs text-slate-500">
                  Your API key is stored securely using system keychain encryption.
                </p>
              </div>
            </div>

            {/* Forward Endpoint */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
                <Server className="w-4 h-4" />
                Forward Endpoint
              </label>
              <input
                type="text"
                value={forwardEndpoint}
                onChange={(e) => setForwardEndpoint(e.target.value)}
                placeholder="https://api.openai.com/v1"
                className="w-full px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-blue-500 focus:outline-none font-mono text-sm"
              />
              <p className="text-xs text-slate-500 mt-1">
                URL of the Privacy Proxy forward endpoint
              </p>
            </div>

            {/* Message */}
            {message && (
              <div
                className={`flex items-center gap-2 p-3 rounded-lg ${
                  message.type === 'success'
                    ? 'bg-green-50 text-green-800 border border-green-200'
                    : 'bg-red-50 text-red-800 border border-red-200'
                }`}
              >
                {message.type === 'success' ? (
                  <CheckCircle2 className="w-5 h-5" />
                ) : (
                  <AlertCircle className="w-5 h-5" />
                )}
                <span className="text-sm">{message.text}</span>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-3 pt-4">
              <button
                onClick={handleSave}
                disabled={isSaving}
                className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium flex-1"
              >
                {isSaving ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-5 h-5" />
                    Save Settings
                  </>
                )}
              </button>
              <button
                onClick={onClose}
                className="px-6 py-3 border-2 border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors font-medium"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

