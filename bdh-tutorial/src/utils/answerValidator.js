/**
 * BDH Tutorial Answer Validator
 * Flexible matching: handles whitespace, common Python equivalences, etc.
 */

/**
 * Normalize an answer for comparison:
 * - Trim whitespace
 * - Collapse internal whitespace
 * - Lowercase
 * - Remove trailing semicolons or commas
 * - Normalize spacing around operators (but NOT around - to avoid breaking negation/subtraction)
 */
function normalize(str) {
  return str
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/[;,]+$/, '')
    .replace(/\s*\(\s*/g, '(')
    .replace(/\s*\)\s*/g, ')')
    .replace(/\s*\*\s*/g, '*')
    .replace(/\s*\+\s*/g, '+')
    .replace(/\s*\/\s*/g, '/')
}

/**
 * Check if user answer matches any accepted answer.
 * Returns { correct: bool, matchType: 'exact'|'normalized'|'none', feedback: string }
 */
export function checkAnswer(userAnswer, acceptedAnswers, blankConfig) {
  if (!userAnswer || userAnswer.trim() === '') {
    return {
      correct: false,
      matchType: 'none',
      feedback: 'Please enter an answer.',
    }
  }

  const userNorm = normalize(userAnswer)

  // Check exact match first
  for (const accepted of acceptedAnswers) {
    if (userAnswer.trim() === accepted.trim()) {
      return {
        correct: true,
        matchType: 'exact',
        feedback: '✓ Correct!',
      }
    }
  }

  // Check normalized match
  for (const accepted of acceptedAnswers) {
    if (userNorm === normalize(accepted)) {
      return {
        correct: true,
        matchType: 'normalized',
        feedback: '✓ Correct! (Minor formatting difference, but logically identical)',
      }
    }
  }

  // NOTE: Partial/contained match removed — it was too permissive and accepted wrong answers
  // that merely contained a substring of the correct answer.

  // Provide a helpful "almost" hint for common mistakes
  const almostHints = getAlmostHints(userAnswer, acceptedAnswers, blankConfig)

  return {
    correct: false,
    matchType: 'none',
    feedback: almostHints || `✗ Not quite. ${blankConfig?.hint ? 'Try the hint for a nudge!' : 'Check the logic again.'}`,
  }
}

function getAlmostHints(userAnswer, acceptedAnswers, blankConfig) {
  const u = userAnswer.trim().toLowerCase()

  // Common Python mistakes
  // NOTE: torch.relu / F.relu equivalence is handled by listing both in acceptedAnswers.
  // No special-case override here to avoid false positives on wrong blanks.

  if (u === 'seq_len * seq_len' || u === 'seq_len**2') {
    if (acceptedAnswers.includes('seq_len')) {
      return '✗ Close! You computed L², but the matrix itself has shape (L, L). The second dimension is just seq_len.'
    }
  }
  if (u === '1024^3') {
    return '✗ Almost! In Python, exponentiation is ** not ^. Try 1024**3.'
  }
  if (u.includes('weight.transpose') && acceptedAnswers.some(a => a.includes('.T'))) {
    return '✓ Correct! .transpose(0,1) and .T are equivalent.'
  }

  return null
}

/**
 * Validate all blanks in an exercise.
 * Returns array of per-blank results.
 */
export function validateExercise(userAnswers, blanks) {
  const results = blanks.map(blank => {
    const userAnswer = userAnswers[blank.id] || ''
    const result = checkAnswer(userAnswer, blank.acceptedAnswers, blank)
    return {
      blankId: blank.id,
      userAnswer,
      ...result,
      explanation: blank.explanation,
    }
  })

  const allCorrect = results.every(r => r.correct)
  const score = results.filter(r => r.correct).length

  return { results, allCorrect, score, total: blanks.length }
}
