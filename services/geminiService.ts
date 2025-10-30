import { GoogleGenAI, Type } from "@google/genai";
import { QuizQuestion, SentimentResult } from '../types';

// Singleton instance, lazily initialized to prevent app crash on load.
let ai: GoogleGenAI | null = null;

/**
 * Lazily initializes and returns the GoogleGenAI instance.
 * This function is called by all other exported functions in this service,
 * ensuring the AI client is only created when first needed.
 * For local development, it falls back to using a key from localStorage.
 * @returns The initialized GoogleGenAI instance.
 * @throws An error if the API key is missing or initialization fails.
 */
const getAiInstance = (): GoogleGenAI => {
    // If already initialized, return the existing instance.
    if (ai) {
        return ai;
    }

    let apiKey: string | null = null;
    
    // 1. Try to get the API key from the environment (for deployed environments)
    try {
        if (typeof process !== 'undefined' && typeof process.env === 'object' && process.env !== null && typeof process.env.API_KEY === 'string' && process.env.API_KEY) {
            apiKey = process.env.API_KEY;
        }
    } catch (e) {
        console.warn("Could not access process.env.API_KEY", e);
    }
    
    // 2. If no environment key, try getting it from localStorage (for local development)
    if (!apiKey) {
        try {
            apiKey = localStorage.getItem('gemini_api_key');
        } catch (e) {
             console.warn("Could not access localStorage", e);
        }
    }

    // 3. If no key is found from any source, throw a specific error for the UI to handle.
    if (!apiKey) {
        console.warn("API key not found in process.env or localStorage.");
        throw new Error("API_KEY_MISSING");
    }

    // Try to create the GoogleGenAI instance.
    try {
        ai = new GoogleGenAI({ apiKey });
        return ai;
    } catch (e) {
        console.error("Could not initialize GoogleGenAI. AI features will be disabled.", e);
        throw new Error("فشل في تهيئة Gemini API. يرجى التحقق من صحة مفتاح API الخاص بك.");
    }
};


const callGeminiWithRetry = async (prompt: string, config: any = {}): Promise<any> => {
    // This will initialize the 'ai' instance on the very first call.
    const aiInstance = getAiInstance();
    const model = 'gemini-2.5-flash';
    let attempts = 3;
    while (attempts > 0) {
        try {
            const response = await aiInstance.models.generateContent({
                model,
                contents: prompt,
                config,
            });
            const textResponse = response.text;

            if (!textResponse || textResponse.trim() === '') {
                 throw new Error("أرجع النموذج استجابة فارغة. قد يكون المحتوى غير واضح أو قصير جدًا للتحليل.");
            }

            if (config.responseMimeType === "application/json") {
                const cleanedJsonString = textResponse.replace(/^```json\s*|```\s*$/g, '').trim();
                if (!cleanedJsonString) {
                    throw new Error("أرجع النموذج استجابة JSON فارغة.");
                }
                try {
                    return JSON.parse(cleanedJsonString);
                } catch(e) {
                    console.error("Failed to parse JSON response:", cleanedJsonString);
                    throw new Error("فشل النموذج في إنشاء استجابة بتنسيق JSON صحيح.");
                }
            }
            return textResponse;
        } catch (error) {
            attempts--;
            console.error(`Gemini API error (attempt ${3 - attempts}):`, error);
            if (attempts === 0) {
                 if (error instanceof Error) {
                    // Re-throw our specific initialization or response errors to be displayed directly in the UI.
                    if (error.message.includes("Gemini API") || error.message.includes("استجابة فارغة") || error.message.includes("JSON") || error.message === "API_KEY_MISSING") {
                        throw error;
                    }
                }
                throw new Error("فشل الاتصال بـ Gemini API بعد عدة محاولات.");
            }
            await new Promise(res => setTimeout(res, 1000));
        }
    }
    throw new Error("فشل الاتصال بـ Gemini API بعد عدة محاولات.");
};


export const analyzeTextContent = async (content: string): Promise<{ analysis: string; categories: string[] }> => {
    if (!content) {
        return { analysis: "الملف فارغ أو لا يمكن قراءة المحتوى.", categories: [] };
    }

    const analysisPrompt = `
        قم بتحليل النص التالي بعناية وقدم ملخصًا شاملاً. يجب أن يحدد الملخص النقاط الرئيسية والحجج والأفكار الأساسية المقدمة في المستند.
        
        يرجى تنظيم إجابتك على النحو التالي:
        1.  **ملخص موجز:** فقرة قصيرة تلخص الغرض العام والمحتوى للمستند.
        2.  **النقاط الرئيسية:** قائمة نقطية (bullet points) لأهم 3-5 نقاط أو نتائج من النص.
        3.  **الاستنتاج:** استنتاج نهائي أو الفكرة الرئيسية التي يمكن استخلاصها من المستند.
        
        تأكد من أن التحليل واضح وموجز وسهل الفهم.
        ---
        محتوى المستند:
        ${content.substring(0, 100000)}
        ---
    `;
    
    // We can run these in parallel to speed up the process
    const analysisPromise = callGeminiWithRetry(analysisPrompt);
    const categoriesPromise = categorizeContent(content);

    const [analysis, categories] = await Promise.all([analysisPromise, categoriesPromise]);

    return { analysis, categories };
};

export const categorizeContent = async (content: string): Promise<string[]> => {
     if (!content) {
        return [];
    }
    const prompt = `
        بناءً على النص التالي، اقترح من 2 إلى 4 تصنيفات ذات صلة. يجب أن يكون كل تصنيف كلمة أو كلمتين فقط.
        مثال على التصنيفات: "فقه", "عقيدة", "تاريخ إسلامي", "سيرة نبوية", "تطوير ذات".
        ---
        محتوى المستند:
        ${content.substring(0, 8000)}
        ---
    `;

    const categoriesSchema = {
        type: Type.ARRAY,
        items: {
            type: Type.STRING,
            description: 'A single category, one or two words max.'
        }
    };

    try {
        const result = await callGeminiWithRetry(prompt, {
            responseMimeType: "application/json",
            responseSchema: categoriesSchema,
        });

        if (Array.isArray(result) && result.every(item => typeof item === 'string')) {
            return result as string[];
        } else {
            console.error("Invalid categories format received from API:", result);
            return []; // Return empty array on failure instead of throwing
        }
    } catch (error) {
        console.error("Failed to generate categories:", error);
        return []; // Return empty on error
    }
};


export async function* summarizeContent(content: string): AsyncGenerator<string> {
    const aiInstance = getAiInstance();
    if (!content) {
        yield "لا يوجد محتوى لتلخيصه.";
        return;
    }
    const prompt = `
        مهمتك هي إنشاء ملخص مفصل للنص التالي. اتبع التعليمات بدقة:
        1.  اقرأ النص بالكامل.
        2.  قم بالمرور على كل فقرة من فقرات النص بشكل منفصل.
        3.  لكل فقرة، اكتب ملخصًا مفصلاً يشرح فكرتها الرئيسية والتفاصيل الهامة.
        4.  يجب أن يكون الناتج النهائي عبارة عن مجموعة من هذه الملخصات المفصلة، واحدة لكل فقرة.
        5.  الهدف هو أن يكون الطول الإجمالي للملخص الناتج كبيرًا، أي ما يقارب نصف طول النص الأصلي. حافظ على وضوح اللغة وسهولة القراءة.
        
        ---
        محتوى المستند:
        ${content.substring(0, 100000)}
        ---
    `;

    const model = 'gemini-2.5-flash';
    try {
        const response = await aiInstance.models.generateContentStream({
            model,
            contents: prompt,
        });

        for await (const chunk of response) {
            const chunkText = chunk.text;
            if (chunkText) {
                yield chunkText;
            }
        }
    } catch (error) {
        console.error(`Gemini API streaming error:`, error);
        if (error instanceof Error) {
            if (error.message.includes("Gemini API") || error.message === "API_KEY_MISSING") {
                throw error;
            }
            throw new Error(`فشل الاتصال بـ Gemini API أثناء التلخيص: ${error.message}`);
        }
        throw new Error("فشل الاتصال بـ Gemini API أثناء التلخيص.");
    }
}

export const createQuiz = async (content: string, questionCount: number): Promise<QuizQuestion[]> => {
    if (!content) {
        return [];
    }
    const prompt = `
        بناءً على النص التالي، قم بإنشاء اختبار من ${questionCount} أسئلة من نوع الاختيار من متعدد. يجب أن يحتوي كل سؤال على 4 خيارات.
        ---
        محتوى المستند:
        ${content.substring(0, 100000)}
        ---
    `;

    const quizSchema = {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            question: {
              type: Type.STRING,
              description: 'The quiz question.',
            },
            options: {
              type: Type.ARRAY,
              items: {
                type: Type.STRING,
              },
              description: 'An array of 4 possible answers.',
            },
            correctAnswerIndex: {
                type: Type.INTEGER,
                description: 'The 0-based index of the correct answer in the options array.',
            },
          },
          required: ["question", "options", "correctAnswerIndex"],
        },
    };

    const result = await callGeminiWithRetry(prompt, {
        responseMimeType: "application/json",
        responseSchema: quizSchema,
    });
    
    if (Array.isArray(result) && result.every(q => 
        typeof q.question === 'string' &&
        Array.isArray(q.options) &&
        q.options.length === 4 &&
        typeof q.correctAnswerIndex === 'number'
    )) {
        return result as QuizQuestion[];
    } else {
        console.error("Invalid quiz format received from API:", result);
        throw new Error("فشل النموذج في إنشاء اختبار بالتنسيق الصحيح.");
    }
};

export const analyzeSentiment = async (content: string): Promise<SentimentResult> => {
    if (!content) {
        throw new Error("لا يوجد محتوى لتحليل المشاعر.");
    }
    const prompt = `
        حلل المشاعر في النص التالي. حدد ما إذا كانت المشاعر العامة "إيجابية" أو "سلبية" أو "محايدة". قدم شرحًا موجزًا لتقييمك.
        ---
        محتوى المستند:
        ${content.substring(0, 100000)}
        ---
    `;

    const sentimentSchema = {
        type: Type.OBJECT,
        properties: {
            sentiment: {
                type: Type.STRING,
                description: 'The overall sentiment, must be one of: "إيجابي", "سلبي", "محايد".'
            },
            explanation: {
                type: Type.STRING,
                description: 'A brief explanation for the sentiment analysis.'
            }
        },
        required: ["sentiment", "explanation"]
    };

    const result = await callGeminiWithRetry(prompt, {
        responseMimeType: "application/json",
        responseSchema: sentimentSchema,
    });

    if (result && typeof result.sentiment === 'string' && typeof result.explanation === 'string') {
        return result as SentimentResult;
    } else {
        console.error("Invalid sentiment format received from API:", result);
        throw new Error("فشل النموذج في تحليل المشاعر بالتنسيق الصحيح.");
    }
};

export const extractKeywords = async (content: string): Promise<string[]> => {
    if (!content) {
        return [];
    }
    const prompt = `
        استنادًا إلى النص التالي، قم باستخراج أهم 5 إلى 10 كلمات رئيسية أو عبارات أساسية تمثل الموضوعات الرئيسية.
        ---
        محتوى المستند:
        ${content.substring(0, 100000)}
        ---
    `;

    const keywordsSchema = {
        type: Type.ARRAY,
        items: {
            type: Type.STRING,
            description: 'A single keyword or key phrase.'
        }
    };

    const result = await callGeminiWithRetry(prompt, {
        responseMimeType: "application/json",
        responseSchema: keywordsSchema,
    });

    if (Array.isArray(result) && result.every(item => typeof item === 'string')) {
        return result as string[];
    } else {
        console.error("Invalid keywords format received from API:", result);
        throw new Error("فشل النموذج في استخراج الكلمات الرئيسية بالتنسيق الصحيح.");
    }
};

export const translateToEnglish = async (content: string): Promise<string> => {
    if (!content) {
        return "لا يوجد محتوى لترجمته.";
    }
    const prompt = `Translate the following Arabic text to English. Maintain the original formatting (markdown headers, lists, bold text, etc.).\n\n---\n\n${content}`;
    return callGeminiWithRetry(prompt);
};

export const translateToArabic = async (content: string): Promise<string> => {
    if (!content) {
        return "لا يوجد محتوى لترجمته.";
    }
    const prompt = `Translate the following English text to Arabic. Maintain the original formatting (markdown headers, lists, bold text, etc.).\n\n---\n\n${content}`;
    return callGeminiWithRetry(prompt);
};

export const generateBookDescription = async (content: string): Promise<string> => {
    if (!content) {
        return "لا يمكن إنشاء وصف لمحتوى فارغ.";
    }
    const prompt = `
        بناءً على مقتطف النص التالي من كتاب، قم بإنشاء وصف جذاب وموجز للكتاب يتكون من حوالي 40 كلمة.
        يجب أن يكون الوصف مناسبًا للعرض في قائمة مكتبة لجذب القارئ.
        ---
        محتوى المستند:
        ${content.substring(0, 4000)} 
        ---
    `;
    return callGeminiWithRetry(prompt);
};

export const generateVideoDescription = async (title: string): Promise<string> => {
    if (!title) {
        return "وصف غير متوفر.";
    }
    const prompt = `
        بناءً على عنوان الفيديو التالي، قم بإنشاء وصف جذاب وموجز للفيديو يتكون من حوالي 30-40 كلمة.
        يجب أن يكون الوصف مناسبًا للعرض في مكتبة وسائط لجذب المشاهد.
        ---
        عنوان الفيديو:
        "${title}"
        ---
    `;
    return callGeminiWithRetry(prompt);
};

export const extractBookTitle = async (content: string): Promise<string> => {
    if (!content) {
        return "عنوان غير معروف";
    }
    const prompt = `
        مهمتك هي التصرف كقارئ ذكي يحلل الصفحة الأولى من كتاب. النص التالي هو بداية مستند تم رفعه. ابحث عن العنوان الرئيسي والبارز في هذا النص، كما لو كان مطبوعًا على غلاف الكتاب.
        
        أريد العنوان فقط، بدون أي كلمات إضافية مثل "العنوان هو:". يجب أن يكون الجواب هو العنوان الصريح.
        
        إذا لم تتمكن من تحديد عنوان واضح، أجب بـ "عنوان غير معروف".
        ---
        مقتطف من الصفحة الأولى:
        ${content.substring(0, 4000)}
        ---
    `;
    return callGeminiWithRetry(prompt);
};

export const generateScriptFromInfo = async (title: string, description: string): Promise<string> => {
    const prompt = `
        مهمتك هي العمل ككاتب سيناريو خبير. بناءً على عنوان الفيديو ووصفه، قم بإنشاء نص تفصيلي (script) للفيديو. 
        يجب أن يكون النص منسقًا بشكل جيد، ويتضمن حوارًا أو حديثًا واضحًا، ويمكن أن يتضمن إشارات للمشاهد المرئية إن أمكن. 
        اجعل النص يبدو طبيعيًا كما لو كان تفريغًا صوتيًا حقيقيًا للفيديو.

        عنوان الفيديو: "${title}"

        وصف الفيديو: "${description}"

        ---
        النص المقترح:
    `;
    return callGeminiWithRetry(prompt);
};

export const generateContentSuggestions = async (content: string): Promise<string[]> => {
    if (!content) {
        return [];
    }
    const prompt = `
        Based on the following Arabic text, provide 5 general, single-word search keywords in Arabic that represent the main themes.
        These keywords should be broad enough to find matches in a library of books and videos.
        Avoid very specific phrases or sentences. Return only a JSON array of single-word strings.
        Example: ["الفقه", "العقيدة", "السنة", "العبادات"]
        ---
        النص:
        ${content.substring(0, 8000)}
        ---
    `;

    const suggestionsSchema = {
        type: Type.ARRAY,
        items: {
            type: Type.STRING,
            description: 'A single suggested topic or title in Arabic.'
        }
    };

    const result = await callGeminiWithRetry(prompt, {
        responseMimeType: "application/json",
        responseSchema: suggestionsSchema,
    });

    if (Array.isArray(result) && result.every(item => typeof item === 'string')) {
        return result as string[];
    } else {
        console.error("Invalid suggestions format received from API:", result);
        throw new Error("فشل النموذج في إنشاء اقتراحات بالتنسيق الصحيح.");
    }
};

export const designArticleFromContent = async (content: string): Promise<string> => {
    if (!content) {
        return "لا يمكن تصميم مقال من محتوى فارغ.";
    }
    const prompt = `
        مهمتك هي العمل ككاتب محتوى محترف. قم بتحويل النص التالي إلى مقال جذاب ومناسب للنشر في مدونة شخصية.
        
        التعليمات:
        1.  ابدأ بعنوان جذاب يلخص الفكرة الرئيسية. استخدم تنسيق Markdown للعنوان (مثال: # عنوان المقال).
        2.  قسم المحتوى إلى فقرات قصيرة وسهلة القراءة.
        3.  استخدم عناوين فرعية (مثال: ## عنوان فرعي) لتنظيم المقال إذا كان المحتوى طويلاً.
        4.  حافظ على جوهر وأفكار النص الأصلي، لكن أعد صياغته بأسلوب شيق ومناسب لجمهور عام.
        5.  في نهاية المقال **بالضبط**، وبدون أي نص إضافي بعدها، أضف السطر التالي: "صمم بواسطة الكُتُبي الذكي لأكاديمية درسني".

        ---
        محتوى المستند الأصلي:
        ${content.substring(0, 100000)}
        ---
    `;
    return callGeminiWithRetry(prompt);
};

export const rateContent = async (content: string): Promise<string> => {
    if (!content) {
        return "لا يوجد محتوى لتقييمه.";
    }
    const prompt = `
        مهمتك هي العمل كناقد أدبي خبير. قم بتقييم النص العربي التالي.
        قدم تقييمًا من 1 إلى 5 نجوم ومراجعة موجزة ومنطقية باللغة العربية تبرر تقييمك.
        خذ في الاعتبار جوانب مثل الوضوح، والبنية، والعمق، والجودة الشاملة.

        ---
        النص للتقييم:
        ${content.substring(0, 100000)}
        ---
    `;

    const ratingSchema = {
        type: Type.OBJECT,
        properties: {
            rating: {
                type: Type.INTEGER,
                description: 'A numerical rating from 1 to 5.'
            },
            review: {
                type: Type.STRING,
                description: 'A concise review in Arabic explaining the rating.'
            }
        },
        required: ["rating", "review"]
    };
    
    try {
        const result = await callGeminiWithRetry(prompt, {
            responseMimeType: "application/json",
            responseSchema: ratingSchema,
        });

        if (result && typeof result.rating === 'number' && typeof result.review === 'string') {
            const rating = Math.max(1, Math.min(5, result.rating)); // Clamp rating between 1 and 5
            const stars = '★'.repeat(rating) + '☆'.repeat(5 - rating);
            return `### تقييم المحتوى\n\n**التقييم:** ${stars} (${rating}/5)\n\n**المراجعة:**\n${result.review}`;
        } else {
            console.error("Invalid rating format received from API:", result);
            throw new Error("فشل النموذج في تقييم المحتوى بالتنسيق الصحيح.");
        }
    } catch (error) {
         console.error("Failed to generate rating:", error);
         throw new Error("حدث خطأ أثناء إنشاء تقييم المحتوى.");
    }
};