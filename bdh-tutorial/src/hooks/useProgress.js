import { useState, useEffect, useCallback } from 'react'

const STORAGE_KEY = 'bdh_tutorial_progress'

const defaultProgress = {
  completedLessons: {},   // { lessonId: true }
  completedExercises: {}, // { exerciseId: { completedAt, blanksCompleted } }
  hints: {},              // { exerciseId_blankId: true }
  lastVisited: null,
}

export function useProgress() {
  const [progress, setProgress] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      return stored ? { ...defaultProgress, ...JSON.parse(stored) } : defaultProgress
    } catch {
      return defaultProgress
    }
  })

  // Persist to localStorage whenever progress changes
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(progress))
    } catch {
      // Storage full or unavailable
    }
  }, [progress])

  const markLessonComplete = useCallback((lessonId) => {
    setProgress(prev => ({
      ...prev,
      completedLessons: { ...prev.completedLessons, [lessonId]: true }
    }))
  }, [])

  const markExerciseComplete = useCallback((exerciseId, blanksCompleted) => {
    setProgress(prev => ({
      ...prev,
      completedExercises: {
        ...prev.completedExercises,
        [exerciseId]: {
          completedAt: new Date().toISOString(),
          blanksCompleted,
        }
      }
    }))
  }, [])

  const useHint = useCallback((exerciseId, blankId) => {
    setProgress(prev => ({
      ...prev,
      hints: { ...prev.hints, [`${exerciseId}_${blankId}`]: true }
    }))
  }, [])

  const setLastVisited = useCallback((chapterSlug, lessonSlug) => {
    setProgress(prev => ({
      ...prev,
      lastVisited: { chapterSlug, lessonSlug, at: new Date().toISOString() }
    }))
  }, [])

  const isLessonComplete = (lessonId) => !!progress.completedLessons[lessonId]
  const isExerciseComplete = (exerciseId) => !!progress.completedExercises[exerciseId]
  const isHintUsed = (exerciseId, blankId) => !!progress.hints[`${exerciseId}_${blankId}`]

  const getTotalCompleted = () => Object.keys(progress.completedLessons).length
  const getTotalExercisesCompleted = () => Object.keys(progress.completedExercises).length

  const resetProgress = () => {
    setProgress(defaultProgress)
    localStorage.removeItem(STORAGE_KEY)
  }

  return {
    progress,
    markLessonComplete,
    markExerciseComplete,
    useHint,
    setLastVisited,
    isLessonComplete,
    isExerciseComplete,
    isHintUsed,
    getTotalCompleted,
    getTotalExercisesCompleted,
    resetProgress,
  }
}
