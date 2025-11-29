// API service for PlainSense backend communication

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Type definitions
export interface LawReference {
  law_name: string;
  section: string;
  article: string;
  full_reference: string;
  explanation: string;
  context: string;
}

export interface ClauseData {
  original: string;
  simplified: string;
  english?: {
    plain: string;
  };
  hindi?: {
    formal: string;
    colloquial: string;
  };
  tamil?: {
    formal: string;
    colloquial: string;
  };
  risk?: {
    level: string;
    explanation: string;
  };
  key_terms?: Array<{
    term: string;
    simplified: string;
  }>;
}

export interface MedicalTestData {
  original: string;
  simplified: string;
  english?: {
    plain: string;
  };
  hindi?: {
    formal: string;
    colloquial: string;
  };
  tamil?: {
    formal: string;
    colloquial: string;
  };
  risk?: {
    level: string;
    explanation: string;
  };
  test_name?: string;
  value?: string;
  normal_range?: string;
  status?: 'high' | 'low' | 'normal';
}

export interface LegalResult {
  success: boolean;
  document_type: string;
  domain: string;
  clauses: ClauseData[];
  law_references?: LawReference[];
  summary?: {
    total_clauses: number;
    high_risk_count: number;
    medium_risk_count: number;
    low_risk_count: number;
    overall_risk?: string;
    recommendations?: string[];
  };
  error?: string;
}

export interface MedicalResult {
  success: boolean;
  document_type: string;
  domain: string;
  results: MedicalTestData[];
  summary?: {
    total_tests: number;
    abnormal_count: number;
    critical_count: number;
    recommendations?: string[];
  };
  error?: string;
}

// Helper function for API calls
async function apiRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      ...options.headers,
    },
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || `API Error: ${response.status}`);
  }

  return data;
}

/**
 * Process a legal document (text or file)
 * @param text Optional text to process
 * @param file Optional file to process
 * @returns Processed clauses with simplifications and translations
 */
export async function processLegalDocument(text?: string, file?: File | null): Promise<LegalResult> {
  if (file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('domain', 'legal');

    return apiRequest<LegalResult>('/process/legal', {
      method: 'POST',
      body: formData,
    });
  } else if (text) {
    return apiRequest<LegalResult>('/process/text', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, domain: 'legal' }),
    });
  } else {
    throw new Error('Please provide text or a file to process');
  }
}

/**
 * Process a medical document (text or file)
 * @param text Optional text to process
 * @param file Optional file to process
 * @returns Processed test results with explanations and translations
 */
export async function processMedicalDocument(text?: string, file?: File | null): Promise<MedicalResult> {
  if (file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('domain', 'medical');

    return apiRequest<MedicalResult>('/process/medical', {
      method: 'POST',
      body: formData,
    });
  } else if (text) {
    return apiRequest<MedicalResult>('/process/text', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, domain: 'medical' }),
    });
  } else {
    throw new Error('Please provide text or a file to process');
  }
}

/**
 * Validate document domain
 * @param file The document file to validate
 * @param expectedDomain The expected domain (legal or medical)
 * @returns Whether the document matches the expected domain
 */
export async function validateDocumentDomain(
  file: File, 
  expectedDomain: 'legal' | 'medical'
): Promise<{ valid: boolean; detected_domain: string; error?: string }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('expected_domain', expectedDomain);

  return apiRequest('/validate/domain', {
    method: 'POST',
    body: formData,
  });
}

/**
 * Get translation for text (loads translation model on demand)
 * @param text Text to translate
 * @param targetLang Target language code ('hi' or 'ta')
 * @returns Translated text
 */
export async function translateText(
  text: string, 
  targetLang: 'hi' | 'ta' | 'hindi' | 'tamil'
): Promise<{ success: boolean; translation: string; source: string; target_lang: string }> {
  return apiRequest('/translate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text, target_lang: targetLang }),
  });
}

/**
 * Health check for API
 * @returns API status
 */
export async function checkApiHealth(): Promise<{ status: string; version: string }> {
  return apiRequest('/health', {
    method: 'GET',
  });
}

const api = {
  processLegalDocument,
  processMedicalDocument,
  validateDocumentDomain,
  translateText,
  checkApiHealth,
};

export default api;
