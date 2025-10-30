// This service uses the pdf.js library, which is expected to be loaded via a script tag in index.html
declare const pdfjsLib: any;

/**
 * Extracts text content from a PDF file located at the given URL.
 * @param url The path or URL to the PDF file.
 * @returns A promise that resolves with the full text content of the PDF.
 */
export const extractTextFromPdf = async (url: string): Promise<string> => {
    // Set the worker source for pdf.js
    pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js`;

    try {
        const loadingTask = pdfjsLib.getDocument(url);
        const pdf = await loadingTask.promise;
        let fullText = '';

        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const textContent = await page.getTextContent();
            // Join all text items on the page with a space
            const pageText = textContent.items.map((item: any) => item.str).join(' ');
            fullText += pageText + '\n\n'; // Add newline between pages for readability
        }

        return fullText;
    } catch (error) {
        console.error('Error reading PDF:', error);
        throw new Error('فشل في قراءة ملف PDF. قد يكون الرابط غير صحيح أو الملف تالفًا.');
    }
};

/**
 * Renders the first page of a PDF to a JPEG Data URL.
 * @param url The path or URL to the PDF file.
 * @returns A promise that resolves with the Base64 Data URL of the rendered page.
 */
export const renderPdfFirstPageToDataUrl = async (url: string): Promise<string> => {
    pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js`;

    try {
        const loadingTask = pdfjsLib.getDocument(url);
        const pdf = await loadingTask.promise;
        const page = await pdf.getPage(1); // Get the first page

        const scale = 1.5;
        const viewport = page.getViewport({ scale });

        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        if (!context) {
            throw new Error('Could not create canvas context');
        }

        canvas.height = viewport.height;
        canvas.width = viewport.width;

        const renderContext = {
            canvasContext: context,
            viewport: viewport,
        };
        
        await page.render(renderContext).promise;
        
        return canvas.toDataURL('image/jpeg', 0.9); // Use JPEG for smaller size
    } catch (error) {
        console.error('Error rendering PDF first page:', error);
        throw new Error('فشل في تحويل الصفحة الأولى من PDF إلى صورة.');
    }
};