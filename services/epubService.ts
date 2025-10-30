// This service uses the ePub.js library, which is expected to be loaded via a script tag in index.html
declare const ePub: any;

/**
 * Extracts text content from an .epub file.
 * @param file The .epub File object.
 * @returns A promise that resolves with the full text content of the EPUB.
 */
export const extractTextFromEpub = async (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async (event) => {
            if (!event.target || !event.target.result) {
                return reject('Failed to read file.');
            }
            try {
                const arrayBuffer = event.target.result as ArrayBuffer;
                const book = ePub(arrayBuffer);
                const navigation = await book.loaded.navigation;
                let fullText = '';

                // Sequentially process each chapter to maintain order
                for (const item of navigation.toc) {
                    const section = book.section(item.href);
                    if (section) {
                        const loadedSection = await section.load();
                        const contents = loadedSection.contents as HTMLElement;
                        // Extract text, trying to maintain paragraph breaks
                        const paragraphs = Array.from(contents.querySelectorAll('p'));
                        if (paragraphs.length > 0) {
                            fullText += paragraphs.map(p => p.textContent).join('\n') + '\n\n';
                        } else {
                            // Fallback for epubs without standard <p> tags
                            fullText += contents.textContent?.replace(/\s+/g, ' ').trim() + '\n\n';
                        }
                    }
                }
                
                book.destroy();
                resolve(fullText);

            } catch (error) {
                console.error('Error processing .epub file:', error);
                reject('فشل في معالجة ملف .epub.');
            }
        };
        reader.onerror = () => {
            reject('Error reading file.');
        };
        reader.readAsArrayBuffer(file);
    });
};