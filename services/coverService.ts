/**
 * Wraps text to fit within a max width on a canvas.
 * @param context The canvas rendering context.
 * @param text The text to wrap.
 * @param x The starting x position.
 * @param y The starting y position.
 * @param maxWidth The maximum width of a line.
 * @param lineHeight The height of a line.
 */
function wrapText(context: CanvasRenderingContext2D, text: string, x: number, y: number, maxWidth: number, lineHeight: number) {
    const words = text.split(' ');
    let line = '';
    for (let n = 0; n < words.length; n++) {
        const testLine = line + words[n] + ' ';
        const metrics = context.measureText(testLine);
        const testWidth = metrics.width;
        if (testWidth > maxWidth && n > 0) {
            context.fillText(line, x, y);
            line = words[n] + ' ';
            y += lineHeight;
        } else {
            line = testLine;
        }
    }
    context.fillText(line, x, y);
}

/**
 * Creates a book cover image from text content.
 * @param text The first few hundred characters of the book.
 * @param title The title of the book.
 * @returns A promise that resolves with a Base64 Data URL of the generated cover image.
 */
export const createCoverFromText = async (text: string, title: string): Promise<string> => {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas');
        const width = 400;
        const height = 600;
        canvas.width = width;
        canvas.height = height;

        const context = canvas.getContext('2d');
        if (!context) {
            reject(new Error('Could not get canvas context'));
            return;
        }

        // Background
        context.fillStyle = '#1e293b'; // slate-800
        context.fillRect(0, 0, width, height);

        // Title
        context.fillStyle = '#f1f5f9'; // slate-100
        context.font = 'bold 36px Arial';
        context.textAlign = 'center';
        wrapText(context, title, width / 2, 80, width - 60, 42);

        // Body text snippet
        context.fillStyle = '#cbd5e1'; // slate-300
        context.font = '18px Arial';
        context.textAlign = 'center';
        
        // Take a snippet of the beginning of the text
        const snippet = text.substring(0, 300) + '...';
        wrapText(context, snippet, width / 2, 200, width - 80, 24);

        resolve(canvas.toDataURL('image/jpeg', 0.9));
    });
};