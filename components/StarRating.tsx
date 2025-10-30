import React, { useState } from 'react';

interface StarRatingProps {
    rating: number;
    onRate?: (newRating: number) => void;
    size?: 'sm' | 'md' | 'lg';
}

const StarRating: React.FC<StarRatingProps> = ({ rating, onRate, size = 'md' }) => {
    const [hoverRating, setHoverRating] = useState(0);

    const isInteractive = !!onRate;

    const handleMouseOver = (index: number) => {
        if (!isInteractive) return;
        setHoverRating(index + 1);
    };

    const handleMouseLeave = () => {
        if (!isInteractive) return;
        setHoverRating(0);
    };

    const handleClick = (index: number) => {
        if (!isInteractive || !onRate) return;
        onRate(index + 1);
    };

    const sizeClasses = {
        sm: 'w-4 h-4',
        md: 'w-6 h-6',
        lg: 'w-8 h-8',
    };

    return (
        <div className="flex items-center" onMouseLeave={isInteractive ? handleMouseLeave : undefined} dir="ltr">
            {[...Array(5)].map((_, index) => {
                // For interactive rating, we just fill or not fill based on hover/current rating.
                if (isInteractive) {
                     const displayRating = hoverRating || rating;
                     const isFilled = displayRating >= index + 1;
                     return (
                        <button
                            key={index}
                            type="button"
                            onClick={() => handleClick(index)}
                            onMouseOver={() => handleMouseOver(index)}
                            className={`p-0 bg-transparent border-none cursor-pointer ${isFilled ? 'text-yellow-400' : 'text-slate-300'} hover:text-yellow-300 transition-colors`}
                            aria-label={`Rate ${index + 1} stars`}
                        >
                            <svg className={sizeClasses[size]} fill="currentColor" viewBox="0 0 20 20">
                                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                            </svg>
                        </button>
                    );
                }

                // For static (non-interactive) rating, we handle partial fills.
                const fillPercentage = Math.max(0, Math.min(1, rating - index)) * 100;
                
                return (
                    <div key={index} className={`relative ${sizeClasses[size]}`}>
                        {/* Background star (empty) */}
                        <svg className="w-full h-full text-slate-300" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                        </svg>
                        {/* Foreground star (filled portion) */}
                        <div className="absolute top-0 left-0 h-full overflow-hidden" style={{ width: `${fillPercentage}%` }}>
                            <svg className={`w-full h-full text-yellow-400`} fill="currentColor" viewBox="0 0 20 20">
                                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                            </svg>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

export default StarRating;