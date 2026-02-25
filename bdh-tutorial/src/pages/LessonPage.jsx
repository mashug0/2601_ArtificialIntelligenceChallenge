import React, { useEffect } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { getLesson, curriculum, getAllLessons } from '../data/curriculum'
import { useProgressContext } from '../App'
import ContentSection from '../components/ContentSection'
import Exercise from '../components/Exercise'
import {
  ChevronLeft, ChevronRight, CheckCircle2, Clock,
  BookOpen, Zap
} from 'lucide-react'
import styles from './LessonPage.module.css'

export default function LessonPage() {
  const { chapterSlug, lessonSlug } = useParams()
  const { markLessonComplete, isLessonComplete, setLastVisited, isExerciseComplete } = useProgressContext()
  const navigate = useNavigate()

  const lessonData = getLesson(chapterSlug, lessonSlug)

  useEffect(() => {
    if (chapterSlug && lessonSlug) {
      setLastVisited(chapterSlug, lessonSlug)
    }
    window.scrollTo(0, 0)
  }, [chapterSlug, lessonSlug])

  if (!lessonData) {
    return (
      <div className={styles.notFound}>
        <h2>Lesson not found</h2>
        <Link to="/">← Back to home</Link>
      </div>
    )
  }

  const { chapter, ...lesson } = lessonData
  const allLessons = getAllLessons()
  const currentIdx = allLessons.findIndex(l => l.id === lesson.id)
  const prevLesson = allLessons[currentIdx - 1]
  const nextLesson = allLessons[currentIdx + 1]

  const lessonDone = isLessonComplete(lesson.id)
  const exerciseDone = lesson.exercise ? isExerciseComplete(lesson.exercise.id) : true

  const handleMarkComplete = () => {
    markLessonComplete(lesson.id)
  }

  return (
    <div className={styles.page}>
      <div className={styles.content}>
        {/* Breadcrumb */}
        <div className={styles.breadcrumb}>
          <Link to="/" className={styles.breadcrumbLink}>{chapter.title}</Link>
          <span className={styles.breadcrumbSep}>/</span>
          <span className={styles.breadcrumbCurrent}>{lesson.title}</span>
        </div>

        {/* Lesson header */}
        <div className={styles.lessonHeader}>
          <div className={styles.lessonMeta}>
            {lesson.estimatedMinutes && (
              <span className={styles.metaItem}>
                <Clock size={13} /> {lesson.estimatedMinutes} min
              </span>
            )}
            {lesson.exercise && (
              <span className={styles.metaItem}>
                <Zap size={13} /> 1 exercise
              </span>
            )}
            {lessonDone && (
              <span className={styles.metaItemDone}>
                <CheckCircle2 size={13} /> Completed
              </span>
            )}
          </div>
          <h1 className={styles.lessonTitle}>{lesson.title}</h1>
        </div>

        {/* Theory sections */}
        <div className={styles.sections}>
          {lesson.sections.map((section, i) => (
            <ContentSection key={i} section={section} />
          ))}
        </div>

        {/* Exercise */}
        {lesson.exercise && (
          <div className={styles.exerciseBlock}>
            <div className={styles.exerciseDivider}>
              <span className={styles.exerciseDividerLabel}>
                <Zap size={14} /> Exercise
              </span>
            </div>
            <Exercise key={lesson.exercise.id} exercise={lesson.exercise} />
          </div>
        )}

        {/* Mark complete button */}
        {!lessonDone && (
          <>
            <button
              className={styles.completeBtn}
              onClick={handleMarkComplete}
              disabled={lesson.exercise && !exerciseDone}
            >
              <CheckCircle2 size={16} />
              Mark as Complete
            </button>
            {lesson.exercise && !exerciseDone && (
              <p className={styles.completeHint}>
                Complete the exercise above to mark this lesson as done.
              </p>
            )}
          </>
        )}

        {lessonDone && (
          <div className={styles.completedNote}>
            <CheckCircle2 size={15} /> Lesson completed!
          </div>
        )}

        {/* Prev / Next navigation */}
        <div className={styles.pagination}>
          {prevLesson ? (
            <Link
              to={`/${prevLesson.chapterSlug}/${prevLesson.slug}`}
              className={styles.pageBtn}
            >
              <ChevronLeft size={16} />
              <div>
                <div className={styles.pageBtnLabel}>Previous</div>
                <div className={styles.pageBtnTitle}>{prevLesson.title}</div>
              </div>
            </Link>
          ) : <div />}

          {nextLesson && (
            <Link
              to={`/${nextLesson.chapterSlug}/${nextLesson.slug}`}
              className={`${styles.pageBtn} ${styles.pageBtnNext}`}
            >
              <div>
                <div className={styles.pageBtnLabel}>Next</div>
                <div className={styles.pageBtnTitle}>{nextLesson.title}</div>
              </div>
              <ChevronRight size={16} />
            </Link>
          )}
        </div>
      </div>
    </div>
  )
}
