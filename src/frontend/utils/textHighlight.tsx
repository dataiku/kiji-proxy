import React from 'react';

interface Entity {
  original: string;
  token: string;
}

interface HighlightRange {
  start: number;
  end: number;
  className: string;
}

/**
 * Safely highlight text by rendering React components instead of using dangerouslySetInnerHTML
 * This prevents XSS vulnerabilities while providing syntax highlighting
 */
export function highlightText(
  text: string,
  ranges: HighlightRange[]
): React.ReactNode {
  if (!text || ranges.length === 0) {
    return text;
  }

  // Sort ranges by start position
  const sortedRanges = [...ranges].sort((a, b) => a.start - b.start);

  const parts: React.ReactNode[] = [];
  let currentIndex = 0;

  sortedRanges.forEach((range, idx) => {
    // Add text before the highlight
    if (currentIndex < range.start) {
      parts.push(
        <span key={`text-${idx}`}>{text.slice(currentIndex, range.start)}</span>
      );
    }

    // Add highlighted text
    parts.push(
      <mark key={`mark-${idx}`} className={range.className}>
        {text.slice(range.start, range.end)}
      </mark>
    );

    currentIndex = range.end;
  });

  // Add remaining text after last highlight
  if (currentIndex < text.length) {
    parts.push(<span key="text-end">{text.slice(currentIndex)}</span>);
  }

  return <>{parts}</>;
}

/**
 * Highlight detected entities in text by their original values
 */
export function highlightEntitiesByOriginal(
  text: string,
  entities: Entity[],
  className: string
): React.ReactNode {
  const ranges: HighlightRange[] = [];

  entities.forEach((entity) => {
    let searchIndex = 0;
    while (searchIndex < text.length) {
      const index = text.indexOf(entity.original, searchIndex);
      if (index === -1) break;

      ranges.push({
        start: index,
        end: index + entity.original.length,
        className,
      });

      searchIndex = index + entity.original.length;
    }
  });

  return highlightText(text, ranges);
}

/**
 * Highlight detected entities in text by their token values
 */
export function highlightEntitiesByToken(
  text: string,
  entities: Entity[],
  className: string
): React.ReactNode {
  const ranges: HighlightRange[] = [];

  entities.forEach((entity) => {
    let searchIndex = 0;
    while (searchIndex < text.length) {
      const index = text.indexOf(entity.token, searchIndex);
      if (index === -1) break;

      ranges.push({
        start: index,
        end: index + entity.token.length,
        className,
      });

      searchIndex = index + entity.token.length;
    }
  });

  return highlightText(text, ranges);
}

/**
 * Optimized function to check if a character index is part of any entity
 * Pre-computes entity ranges for O(1) lookups instead of O(n) per character
 */
export function createEntityRangeChecker(
  text: string,
  entities: Entity[]
): (index: number) => boolean {
  const rangeSet = new Set<number>();

  entities.forEach((entity) => {
    let searchIndex = 0;
    while (searchIndex < text.length) {
      const index = text.indexOf(entity.original, searchIndex);
      if (index === -1) break;

      // Mark all indices in this range
      for (let i = index; i < index + entity.original.length; i++) {
        rangeSet.add(i);
      }

      searchIndex = index + entity.original.length;
    }
  });

  return (index: number) => rangeSet.has(index);
}

/**
 * Render text with character-level highlighting for diff view
 * More efficient than the original implementation
 */
export function highlightTextByCharacter(
  text: string,
  entities: Entity[],
  highlightClassName: string
): React.ReactNode {
  const isHighlighted = createEntityRangeChecker(text, entities);

  const chars = text.split('');
  const parts: React.ReactNode[] = [];
  let currentSpan: string[] = [];
  let isCurrentHighlighted = false;

  chars.forEach((char, idx) => {
    const shouldHighlight = isHighlighted(idx);

    if (shouldHighlight === isCurrentHighlighted) {
      // Continue current span
      currentSpan.push(char);
    } else {
      // Flush current span and start new one
      if (currentSpan.length > 0) {
        parts.push(
          isCurrentHighlighted ? (
            <span key={idx} className={highlightClassName}>
              {currentSpan.join('')}
            </span>
          ) : (
            <span key={idx}>{currentSpan.join('')}</span>
          )
        );
      }
      currentSpan = [char];
      isCurrentHighlighted = shouldHighlight;
    }
  });

  // Flush remaining span
  if (currentSpan.length > 0) {
    parts.push(
      isCurrentHighlighted ? (
        <span key="end" className={highlightClassName}>
          {currentSpan.join('')}
        </span>
      ) : (
        <span key="end">{currentSpan.join('')}</span>
      )
    );
  }

  return <>{parts}</>;
}
