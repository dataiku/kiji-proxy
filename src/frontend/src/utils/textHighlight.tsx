/**
 * Text highlighting utility using HTML string generation instead of React components
 * This approach eliminates memory overhead from creating thousands of React components
 * by generating HTML strings once and injecting them via dangerouslySetInnerHTML
 */

interface Entity {
  original: string;
  token: string;
}

// HTML escape utility to prevent XSS
function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Highlight text by generating an HTML string with <mark> elements
 * This creates zero React components, preventing memory issues
 *
 * @param text - The text to highlight
 * @param highlights - Array of {text, className} to highlight
 * @returns HTML string with <mark> tags
 */
function highlightTextToHTML(
  text: string,
  highlights: Array<{ text: string; className: string }>
): string {
  if (!text || highlights.length === 0) {
    return escapeHtml(text);
  }

  // Safety limit: truncate very large text
  const MAX_TEXT_SIZE = 100000; // 100KB
  if (text.length > MAX_TEXT_SIZE) {
    console.warn(
      `[SAFETY] Text size ${text.length} exceeds limit ${MAX_TEXT_SIZE}, truncating`
    );
    text =
      text.substring(0, MAX_TEXT_SIZE) + "\n\n... [Text truncated for safety]";
  }

  // Build a list of all occurrences to highlight
  interface Range {
    start: number;
    end: number;
    className: string;
  }

  const ranges: Range[] = [];

  // Find all occurrences of each highlight text
  highlights.forEach(({ text: searchText, className }) => {
    let searchIndex = 0;
    let foundCount = 0;
    const MAX_OCCURRENCES = 500; // Limit occurrences per search text

    while (searchIndex < text.length && foundCount < MAX_OCCURRENCES) {
      const index = text.indexOf(searchText, searchIndex);
      if (index === -1) break;

      ranges.push({
        start: index,
        end: index + searchText.length,
        className,
      });

      foundCount++;
      searchIndex = index + searchText.length;
    }

    if (foundCount >= MAX_OCCURRENCES) {
      console.warn(
        `[SAFETY] Found ${foundCount}+ occurrences of "${searchText}", limiting to ${MAX_OCCURRENCES}`
      );
    }
  });

  // Sort ranges by start position
  ranges.sort((a, b) => a.start - b.start);

  // Merge overlapping ranges to avoid nested <mark> tags
  const mergedRanges: Range[] = [];
  if (ranges.length > 0) {
    let current = ranges[0];

    for (let i = 1; i < ranges.length; i++) {
      if (ranges[i].start <= current.end) {
        // Overlapping - extend current range
        current = {
          start: current.start,
          end: Math.max(current.end, ranges[i].end),
          className: current.className, // Keep first className
        };
      } else {
        mergedRanges.push(current);
        current = ranges[i];
      }
    }
    mergedRanges.push(current);
  }

  // Build HTML string with highlights
  let html = "";
  let currentIndex = 0;

  mergedRanges.forEach((range) => {
    // Add escaped text before highlight
    if (currentIndex < range.start) {
      html += escapeHtml(text.slice(currentIndex, range.start));
    }

    // Add highlighted text with <mark> tag
    html += `<mark class="${range.className}">${escapeHtml(
      text.slice(range.start, range.end)
    )}</mark>`;

    currentIndex = range.end;
  });

  // Add remaining text after last highlight
  if (currentIndex < text.length) {
    html += escapeHtml(text.slice(currentIndex));
  }

  return html;
}

/**
 * Highlight entities by their original text values
 * Returns HTML string for use with dangerouslySetInnerHTML
 */
export function highlightEntitiesByOriginal(
  text: string,
  entities: Entity[],
  className: string
): string {
  // Safety check: limit entity count
  const MAX_ENTITIES = 500;
  if (entities.length > MAX_ENTITIES) {
    console.warn(
      `[SAFETY] Entity count ${entities.length} exceeds limit ${MAX_ENTITIES}, limiting`
    );
    entities = entities.slice(0, MAX_ENTITIES);
  }

  const highlights = entities.map((entity) => ({
    text: entity.original,
    className,
  }));

  return highlightTextToHTML(text, highlights);
}

/**
 * Highlight entities by their token values
 * Returns HTML string for use with dangerouslySetInnerHTML
 */
export function highlightEntitiesByToken(
  text: string,
  entities: Entity[],
  className: string
): string {
  // Safety check: limit entity count
  const MAX_ENTITIES = 500;
  if (entities.length > MAX_ENTITIES) {
    console.warn(
      `[SAFETY] Entity count ${entities.length} exceeds limit ${MAX_ENTITIES}, limiting`
    );
    entities = entities.slice(0, MAX_ENTITIES);
  }

  const highlights = entities.map((entity) => ({
    text: entity.token,
    className,
  }));

  return highlightTextToHTML(text, highlights);
}

/**
 * Legacy function name for backwards compatibility
 * Uses highlightEntitiesByOriginal internally
 */
export function highlightTextByCharacter(
  text: string,
  entities: Entity[],
  className: string
): string {
  return highlightEntitiesByOriginal(text, entities, className);
}

/**
 * Generic highlight function for custom text patterns
 * Returns HTML string for use with dangerouslySetInnerHTML
 */
export function highlightText(
  text: string,
  searchTexts: string[],
  className: string
): string {
  const highlights = searchTexts.map((searchText) => ({
    text: searchText,
    className,
  }));

  return highlightTextToHTML(text, highlights);
}
