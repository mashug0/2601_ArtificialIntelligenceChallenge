import React, { useState } from 'react'
import Editor from '@monaco-editor/react'
import { validateExercise } from '../utils/answerValidator'
import { useProgressContext } from '../App'
import {
  Lightbulb, CheckCircle2, XCircle, RotateCcw,
  ChevronDown, ChevronUp, PlayCircle, Trophy, Eye
} from 'lucide-react'
import styles from './Exercise.module.css'

export default function Exercise({ exercise }) {
  const { markExerciseComplete, isExerciseComplete, useHint } = useProgressContext()

  const [code, setCode] = useState(exercise.starterCode)
  const [validationResult, setValidationResult] = useState(null)
  const [showHints, setShowHints] = useState({})
  const [celebrateAnim, setCelebrateAnim] = useState(false)
  const [showExpected, setShowExpected] = useState(false)

  const alreadyDone = isExerciseComplete(exercise.id)

  const handleCodeChange = (newCode) => {
    setCode(newCode)
    setValidationResult(null)
  }

  const handleCheckCode = () => {
    const answers = {}
    const userLines = code.split('\n')

    exercise.blanks.forEach(blank => {
      // Split the starter code on this blank's placeholder to get surrounding context
      const parts = exercise.starterCode.split(blank.placeholder)
      if (parts.length < 2) {
        answers[blank.id] = ''
        return
      }

      // Get the partial line text immediately before and after the blank
      const beforeLines = parts[0].split('\n')
      const afterLines  = parts[1].split('\n')
      const linePrefix  = beforeLines[beforeLines.length - 1] // text before blank on same line
      const lineSuffix  = afterLines[0]                       // text after blank on same line

      // Find the user's line that starts with the same prefix
      for (const line of userLines) {
        if (linePrefix === '' ? true : line.startsWith(linePrefix)) {
          let answer = line.slice(linePrefix.length)
          // Trim off everything from the suffix onward (rest of line after the blank)
          if (lineSuffix && lineSuffix.trim() !== '' && answer.includes(lineSuffix)) {
            answer = answer.slice(0, answer.indexOf(lineSuffix))
          }
          // Strip inline comments and trim whitespace
          answer = answer.split('#')[0].trim()
          if (answer !== '') {
            answers[blank.id] = answer
            break
          }
        }
      }

      if (!answers[blank.id]) answers[blank.id] = ''
    })

    const validation = validateExercise(answers, exercise.blanks)
    setValidationResult(validation)

    if (validation.allCorrect) {
      markExerciseComplete(exercise.id, validation.score)
      setCelebrateAnim(true)
      setTimeout(() => setCelebrateAnim(false), 2000)
    }
  }

  const handleReset = () => {
    setCode(exercise.starterCode)
    setValidationResult(null)
    setShowHints({})
    setShowExpected(false)
  }

  const handleHint = (blankId) => {
    setShowHints(prev => ({ ...prev, [blankId]: true }))
    useHint(exercise.id, blankId)
  }

  const difficultyColor = {
    beginner: '#4ade80',
    intermediate: '#ffb347',
    advanced: '#f87171',
  }

  return (
    <div className={`${styles.exercise} ${celebrateAnim ? styles.celebrate : ''}`}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.badge} style={{ color: difficultyColor[exercise.difficulty] }}>
            {exercise.difficulty}
          </div>
          <h3 className={styles.title}>{exercise.title}</h3>
        </div>
        {alreadyDone && (
          <div className={styles.completedBadge}>
            <Trophy size={14} /> Complete
          </div>
        )}
      </div>

      <p className={styles.instructions}>{exercise.instructions}</p>

      {/* Main content: Editor + Hints Sidebar */}
      <div className={styles.editorContainer}>
        {/* Left: Code Editor */}
        <div className={styles.editorMain}>
          <div className={styles.editorHeader}>
            <span className={styles.editorLang}>Python</span>
            <span className={styles.editorNote}>Replace each ___BLANK_N___ placeholder with your answer</span>
          </div>
          <div className={styles.monacoWrapper}>
            <Editor
              height="400px"
              language="python"
              value={code}
              onChange={handleCodeChange}
              options={{
                readOnly: alreadyDone,
                minimap: { enabled: false },
                fontSize: 13,
                fontFamily: "'JetBrains Mono', monospace",
                lineHeight: 20,
                padding: { top: 16, bottom: 16 },
                scrollBeyondLastLine: false,
                renderLineHighlight: 'none',
                overviewRulerLanes: 0,
                scrollbar: { vertical: 'auto', horizontal: 'auto' },
                wordWrap: 'on',
              }}
              theme="vs-dark"
            />
          </div>
        </div>

        {/* Right: Hints Sidebar */}
        <div className={styles.hintsSidebar}>
          <div className={styles.hintsHeader}>
            <Lightbulb size={14} /> Hints
          </div>
          <div className={styles.hintsList}>
            {exercise.blanks.map((blank, idx) => (
              <div key={blank.id} className={styles.hintItem}>
                <div className={styles.hintNumber}>Step {idx + 1}</div>
                <code className={styles.hintPlaceholder}>{blank.placeholder}</code>
                <button
                  className={styles.hintToggleBtn}
                  onClick={() => handleHint(blank.id)}
                >
                  {showHints[blank.id] ? 'Hide' : 'Show'} hint
                </button>
                {showHints[blank.id] && (
                  <div className={styles.hintContent}>{blank.hint}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Validation Results */}
      {validationResult && (
        <div className={`${styles.validationResult} ${validationResult.allCorrect ? styles.validationSuccess : styles.validationError}`}>
          <div className={styles.validationHeader}>
            {validationResult.allCorrect ? (
              <>
                <CheckCircle2 size={18} /> Great! All answers are correct.
              </>
            ) : (
              <>
                <XCircle size={18} /> Some answers need adjustment.
              </>
            )}
          </div>
          {validationResult.results && (
            <div className={styles.resultsList}>
              {validationResult.results.map((result, idx) => (
                <div key={idx} className={`${styles.resultItem} ${result.correct ? styles.resultCorrect : styles.resultWrong}`}>
                  <div className={styles.resultDot}>{result.correct ? <CheckCircle2 size={14} /> : <XCircle size={14} />}</div>
                  <div className={styles.resultContent}>
                    <div className={styles.resultText}>{result.feedback}</div>
                    {result.correct && result.explanation && (
                      <div className={styles.resultExplanation}>{result.explanation}</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Expected output — hidden behind toggle */}
      {exercise.expectedOutput && (
        <div className={styles.expectedOutput}>
          <button
            className={styles.expectedToggle}
            onClick={() => setShowExpected(s => !s)}
          >
            <Eye size={13} /> {showExpected ? 'Hide' : 'Show'} expected output
          </button>
          {showExpected && (
            <pre className={styles.expectedCode}>{exercise.expectedOutput}</pre>
          )}
        </div>
      )}

      {/* Actions */}
      <div className={styles.inlineActions}>
        <button
          className={`${styles.checkBtn} ${alreadyDone ? styles.checkBtnDisabled : ''}`}
          onClick={handleCheckCode}
          disabled={alreadyDone}
        >
          <PlayCircle size={14} /> Check Code
        </button>
        <button className={styles.resetBtn} onClick={handleReset}>
          <RotateCcw size={13} /> Reset
        </button>
      </div>

      {/* Completion banners — separate cases for revisit vs new completion */}
      {alreadyDone && !validationResult && (
        <div className={`${styles.resultBanner} ${styles.resultBannerGood}`}>
          <Trophy size={16} />
          <strong>Already completed!</strong>
          {' '}You finished this exercise in a previous session.
        </div>
      )}
      {validationResult?.allCorrect && (
        <div className={`${styles.resultBanner} ${styles.resultBannerGood}`}>
          <Trophy size={16} />
          <strong>Exercise Complete!</strong>
          {' '}You completed all {exercise.blanks.length} steps correctly.
        </div>
      )}
    </div>
  )
}
