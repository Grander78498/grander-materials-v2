#set text(font: "Times New Roman", size: 14pt, lang: "ru")
#set page(margin: (
  top: 2cm,
  left: 3cm,
  right: 1cm,
  bottom: 2cm
))

#let quote = symbol(
  ("left", "«"),
  ("right", "»")
)

#let quoted(it) = {
  quote.left + it + quote.right
}

#let titul(..it) = {
  
  figure(
    image("/src/logo.jpg", height: 13%),
  )
  set align(center)
  set par(leading: 6pt, spacing: 6pt)
  "МИНОБРНАУКИ РОССИИ
  Федеральное государственное бюджетное образовательное учреждение высшего образования\n"
  
  text(weight: "black", quoted("МИРЭА - Российский технологический университет") + "\nРТУ МИРЭА")
  line(length: 95%)
  v(-3.5pt)
  line(length: 95%)
  
  linebreak()
  linebreak()
  
  
  text(weight: "black", "Институт ") 
  it.named().Институт
  linebreak()
  
  text(weight: "black", "Кафедра ") 
  it.named().Кафедра
  
  linebreak()
  linebreak()
  
  text(weight: "black", it.named().Практика)

  linebreak()

  text(weight: "black", "\nпо дисциплине\n" + it.named().Дисциплина)
  
  v(60pt)

  align([*Выполнили:*], left)
  
  grid(columns: (65%, 35%), align:(left, center), stroke: none, inset: (y: 3pt),
  grid.cell(rowspan: 2, [Студенты группы #it.named().Группа]),
  [#it.named().Студент])

  v(36pt)
  
  grid(columns: (65%, 35%), align:(left, center), stroke: none, inset: (y: 3pt),
  grid.cell(rowspan: 2, [*Проверил* #it.named().at("Должность", default: "")]),
  [#it.named().Преподаватель])
  
  align(bottom, [Москва 2024])
}