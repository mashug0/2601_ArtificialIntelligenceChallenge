import React, { createContext, useContext } from 'react'
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useProgress } from './hooks/useProgress'
import Layout from './components/Layout'
import LessonPage from './pages/LessonPage'
import HomePage from './pages/HomePage'

// Global Progress Context
export const ProgressContext = createContext(null)
export const useProgressContext = () => useContext(ProgressContext)

export default function App() {
  const progressData = useProgress()

  return (
    <ProgressContext.Provider value={progressData}>
      <HashRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<HomePage />} />
            <Route path=":chapterSlug/:lessonSlug" element={<LessonPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </HashRouter>
    </ProgressContext.Provider>
  )
}
