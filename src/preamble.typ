#let main(doc) = {
  set text(font: "Times New Roman", size: 14pt, lang: "ru", hyphenate: false)
  set page(margin: (
    top: 2cm,
    left: 3cm,
    right: 1cm,
    bottom: 2cm
  ))
  doc
}

#let print_symbols(..args) = {
  let res = []
  for (i, arg) in args.pos().enumerate() {
    res += arg;
    if i == args.pos().len() - 1 {
      res += "."
    }
    else {
      res += ";"
    }
    res += linebreak()
  }
  set par(first-line-indent: (amount: 1.25cm, all: true), hanging-indent: 1.25cm)
  [
    #place(left, [где])
    #par(res)
  ]
}

#let appendix-num(.., last) = {
  let symbols = "абвгдежиклмнпрстуфхцшщэюя".split("").map(upper)
  symbols.at(last)
}

#let eq-simple(eq) = {
  set par(leading: 8pt)
  set block(breakable: true)
  math.equation(eq, block: true, numbering: none)
}

#let appendix() = {
  outline(target: heading.where(level: 6), title: heading("Приложения", level: 1, outlined: true))
  counter(heading.where(level: 6)).update(0)
}


#let template(doc) = {
  set par(first-line-indent: (
    amount: 1.25cm,
    all: true
  ), justify: true, leading: 14pt)
  show raw: it => {
    set par(leading: 4pt)
    set text(size: 10pt, font: "Courier New")
    it
  }
  set page(footer: context[
    #let (num,) = counter(page).get()
    #if num > 1 {
      align(center, counter(page).display(
        "1"
      ))
    }
  ])
  show figure.where(kind: raw): it => {
    set align(left)
    set text(font: "Courier New", size: 10pt)
    set figure(numbering: "A 1")
    set block(breakable: true)
    set figure.caption(position: top)
    show figure.caption: set text(size: 12pt)
    par([#it.caption #rect(width: 100%, stroke: luma(50%), [#it.body])])
  }
  show heading: it => {
    let res
    if it.numbering != none {
      context {
        let arr = counter(heading).get()
        let res
        if it.numbering == none {
          set par(justify: true)
          res = align(center, it.body)
        } else {
          set par(first-line-indent: (amount: 1.25cm, all: true), justify: true)
          res = [#arr.map(str).join(".") #it.body]
        }

        block(par[#res], sticky: true)
      }
    }
    else {
      align(center, it)
    }
    v(12pt)
  }
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => {
    counter(math.equation).update(0)
    counter(figure.where(kind: image)).update(0)
    if it.body != [Приложения] {
      pagebreak()
    }
    upper(text(it, size: 18pt))
  }
  show heading.where(level: 2): it => {
    v(24pt, weak: true)
    counter(math.equation).update(0)
    counter(figure.where(kind: image)).update(0)
    text(it, size: 16pt)
  }
  show heading.where(level: 3): it => {
    v(24pt, weak: true)
    it
  }
  show heading.where(level: 6): it => {
    pagebreak()
    counter(heading.where(level: 6)).step()
    counter("code").update(0)
    align(center, "Приложение " + appendix-num(counter(heading.where(level: 6)).get().first() + 1))
    v(12pt)
    align(center, text(it.body, weight: "regular"))
  }

  show heading: set heading(supplement: none)
  show heading.where(numbering: none): it => {
    align(center, it)
  }

  set list(indent: 1.25cm, body-indent: 1cm)
  show list: it => {
    set par(first-line-indent: (amount: 1.25cm, all: true), hanging-indent: 2.25cm)
    for (index, item) in it.children.enumerate() {
      let res = it.marker.at(0) + h(1cm - 4pt) + item.body;
      if index == it.children.len() - 1 {
        res += ".";
      }
      else {
        res += ";";
      }
      par(res)
    }
  }

  set enum(indent: 1.25cm, body-indent: 1cm, full: true)
  
  show outline.entry: it => {
    set par(justify: true)
    if it.level == 1 {
      upper(it)
    }
    else if it.level == 2 or it.level == 3 {
      it
    }
    else {
      context {
        let app-num = counter(heading.where(level: 6)).get().first()
        counter(heading.where(level: 6)).step()
        par(link(query(heading.where(level: 6)).at(app-num).location())[Приложение #appendix-num(app-num + 1) --- #it.element.body.])
      }
    }
  }
  set outline(indent: 0cm)
  show outline: set outline(title: align(center, "Содержание"), 
                            target: selector(heading.where(level: 1))
                            .or(heading.where(level: 2))
                            .or(heading.where(level: 3)))
  show outline: it => {
    it
    counter(heading.where(level: 1)).update(0)
  }
  show math.equation: set block(breakable: true)
  set math.equation(numbering: num => {
    let arr = counter(heading).get()
    if arr.len() >= 2 {
      numbering("(1.1.1)", ..arr.slice(0, 2), num)
    }
    else {
      numbering("(1.1.1)", ..arr, num)
    }
  }, supplement: none)
  set figure(
    supplement: none
  )
  show figure.where(kind: image): it => {
    let arr = counter(heading).get()
    it.body
    set text(size: 12pt, weight: "black")
    set par(leading: 4pt, justify: false)
    v(-12pt)
    [
      Рисунок #arr.slice(0, 2).map(str).join(".").#counter(figure.where(kind: image)).get().first() --- #it.caption.body
    ]
    v(-6pt)
  }
  show figure.where(kind: table): it => {
    it
  }
  show figure.where(kind: table, supplement: [Листинг]): it => {
    it.body
  }
  show ref: it => {
    if it.element != none and it.element.func() == math.equation {
      show regex("\("): _ => {}
      show regex("\)"): _ => {}
      link(it.target)[#it]
    } else if it.element != none and it.element.supplement == [Листинг] {
      context {
        link(it.target)[#appendix-num(counter(heading.where(level: 6)).at(it.target).first()).#counter("code").at(it.target).first()]
      }
    } else if it.element != none and it.element.body.func() == image {
      context {
        let arr = counter(heading).at(it.target)
        link(it.target)[#arr.slice(0, 2).map(str).join(".").#counter(figure.where(kind: image)).at(it.target).first()]
      }
    } else if it.element != none and it.element.body.func() == table {
      context {
        let arr = counter(heading).at(it.target)
        link(it.target)[#arr.slice(0, 2).map(str).join(".").#counter(figure.where(kind: table)).at(it.target).first()]
      }
    } else {
      it
    }
  }
  show math.equation.where(block: true): it => {
    set block(breakable: true)
    set text(font: "Cambria Math")
    linebreak()
    it
    linebreak()
  }
  
  show bibliography: set block(width: 100%)
  doc
}


#let next-page-code(next-page-content: [], label-name, ..table-args) = context {
  show figure.where(kind: table): set block(breakable: true)
  show figure.where(kind: table): set figure(caption: none)
  show figure.where(kind: table): it => {
    align(left, it);
  }
  let columns = table-args.named().at("columns", default: 1)
  let column-amount = if type(columns) == int {
    columns
  } else if type(columns) == array {
    columns.len()
  } else {
    1
  }

  // Counter of tables so we can create a unique table-part-counter for each table
  let table-counter = counter("code")
  table-counter.step()

  // Counter for the amount of pages in the table
  // It is increased by one for each footer repetition
  let table-part-counter = counter("table-part" + str(counter(heading).get().at(3)) + str(table-counter.get().first()))
  show <table-header>: _ => {
    table-part-counter.step()
    set text(font: "Times New Roman", size: 12pt)
    context {
      if table-part-counter.get().first() == 1 {
        v(0.5em)
        emph(next-page-content.at(0))
      }
      else if table-part-counter.get() != table-part-counter.final() {
        v(-0.5em)
        emph(next-page-content.at(1))
      }
      else {
        v(-0.5em)
        emph(next-page-content.last())
      }
    }
  }
  [#figure(table(
    table.header(
      // The 'next page' content spans all columns and has no stroke
      // Must be selectable by the show rule above which hides it at the last page
      table.cell(colspan: column-amount, stroke: none, [#next-page-content <table-header>])
    ),
    columns: (1fr, ),
    ..table-args,
    stroke: luma(50%),
  ), supplement: [Листинг]) #if label-name != none {label-name}]

  // Compensate for the empty footer at the last page of the table
  v(-measure(next-page-content.last()).height)
}

#let next-page-table(next-page-content: [], label-name, ..table-args) = context {
  set text(size: 12pt)
  show figure.where(kind: table): set block(breakable: true)
  show figure.where(kind: table): set figure(caption: none)
  show figure.where(kind: table): it => {
    align(left, it);
  }

  let columns = table-args.named().at("columns", default: 1)
  let column-amount = if type(columns) == int {
    columns
  } else if type(columns) == array {
    columns.len()
  } else {
    1
  }

  let table-counter = counter("table")
  table-counter.step()

  let arr = counter(heading).get()
  let table-part-counter = counter("table-part" + str(arr.slice(0, 1).map(str).join(".")) + str(table-counter.get().first()))
  show <table-header>: _ => {
    table-part-counter.step()
    set text(font: "Times New Roman", size: 12pt)
    context {
      if table-part-counter.get().first() == 1 {
        emph(next-page-content.at(0))
      }
      else if table-part-counter.get() != table-part-counter.final() {
        v(-6pt)
        emph(next-page-content.at(1))
      }
      else {
        v(-6pt)
        emph(next-page-content.last())
      }
    }
  }
  [#figure(table(
    table.header(table.cell(table(
      table.cell(colspan: 1, stroke: none, inset: (bottom: -6pt, top: 0pt, left: -6pt), align: left, [#next-page-content <table-header>])
    ), stroke: none)),
    columns: (1fr),
    table.cell(..table-args, stroke: none, align: center),
  ), supplement: [Таблица]) #if label-name != none {label-name}]
  v(-measure(next-page-content.last()).height)
  v(-6pt)
}

#let simple-code(code-content, name, label: none) = {
  set par(first-line-indent: 0cm)
  context [
    #let num = appendix-num(counter(heading).get().at(3)) + "." + str(counter("code").get().first() + 1)
  #next-page-code(
    next-page-content: (par([#h(-4pt) Листинг #num --- #name], leading: 4pt), [Продолжение Листинга #num], [Окончание Листинга #num]),
    label,
    code-content
  )
  ]
}

#let simple-table(columns: none, name: "", label: none, header: none, ..table-content) = {
  v(-6pt)
  set par(first-line-indent: 0cm, justify: false)
  set text(hyphenate: true)
  set enum(body-indent: 6pt, indent: 0pt)
  context [
    #let arr = counter(heading).get()
    #let num = arr.slice(0, 1).map(str).join(".") + "." + str(counter("table").get().first() + 1)
  #next-page-table(
    next-page-content: (par([Таблица #num --- #name], leading: 4pt), [Продолжение Таблицы #num]),
    label,
    table(align: left, columns: if columns == none {header.len()} else {columns}, table.header(..header.map(text.with(weight: "bold")).map(align.with(center)), repeat: false), ..table-content, stroke: luma(100))
  )
  ]
}