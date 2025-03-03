#import "/src/preamble.typ": *
#import "/src/titul-holm.typ": *
#show: main
#titul(
  Институт: [Информационных Технологий],
  Кафедра: [Вычислительной техники],
  Практика: upper[Практическая работа №1],
  Дисциплина: ["Проектирование интеллектуальных систем (Часть 1/2)"],
  Группа: [ИКБО-04-22],
  Студент: [Егоров Л.А.],
  Преподаватель: [Холмогоров В.В.]
)
#show: template

#outline()

#heading([Введение], numbering: none)
Какое-то введение

= Теоретическая часть
== Алгоритм Apriori
== Алгоритм Eclat
== Алгоритм FP-Growth

= Описание данных
== Генерация данных

= Практическая часть
== Реализация алгоритма Apriori
== Реализация алгоритма Eclat
== Реализация алгоритма FP-Growth

#heading([Заключение], numbering: none)
Какое-то заключение

#bibliography("authors1.bib", full: true, style: "/src/gost-r-7-0-5-2008-numeric-alphabetical.csl", title: "Список использованных источников")

#appendix()

#set heading(offset: 5)
= Реализация алгоритма Apriori
= Реализация алгоритма Eclat
= Реализация алгоритма FP-Growth