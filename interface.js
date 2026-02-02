export class Interface {
  constructor(canvas, callbacks) {
    this.canvas = canvas
    this.onSwipe = callbacks.onSwipe
    this.onDrag = callbacks.onDrag
    this.isDragging = false
    this.swipedInCurrentStroke = false
    this.startNode = null
    this.padding = 60
    this.accumX = 0
    this.accumY = 0
    this.wheelThreshold = 80
    this.wheelResetTimer = null
    this.canvas.addEventListener('pointerdown', this.onDown.bind(this))
    window.addEventListener('pointermove', this.onMove.bind(this))
    window.addEventListener('pointerup', this.onUp.bind(this))
    window.addEventListener('wheel', this.onWheel.bind(this), { passive: false })
  }

  getGridNode(x, y) {
    const w = this.canvas.width
    const h = this.canvas.height
    const usableW = w - (this.padding * 2)
    const usableH = h - (this.padding * 2)  
    const col = Math.min(2, Math.max(0, Math.floor((x - this.padding) / (usableW / 3))))
    const row = Math.min(2, Math.max(0, Math.floor((y - this.padding) / (usableH / 3))))
    return { 
      col, row,
      centerX: this.padding + (col + 0.5) * (usableW / 3),
      centerY: this.padding + (row + 0.5) * (usableH / 3)
    }
  }

  onDown(e) {
    const rect = this.canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) * (this.canvas.width / rect.width)
    const y = (e.clientY - rect.top) * (this.canvas.height / rect.height)
    this.isDragging = true
    this.swipedInCurrentStroke = false
    this.startNode = this.getGridNode(x, y) 
    if (this.onDrag) {
      this.onDrag({ active: true, startX: this.startNode.centerX, startY: this.startNode.centerY, currX: x, currY: y })
    }
  }

  onMove(e) {
    if (!this.isDragging || this.swipedInCurrentStroke) return
    const rect = this.canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) * (this.canvas.width / rect.width)
    const y = (e.clientY - rect.top) * (this.canvas.height / rect.height)
    const dist = Math.hypot(x - this.startNode.centerX, y - this.startNode.centerY)
    if (dist < 50) return
    const currentNode = this.getGridNode(x, y)
    if (currentNode.col !== this.startNode.col || currentNode.row !== this.startNode.row) {
      const type = this.resolveVector(currentNode.col - this.startNode.col, currentNode.row - this.startNode.row)
      if (type !== 'unknown') {
        this.swipedInCurrentStroke = true
        if (this.onSwipe) this.onSwipe(type)
      }
    }
    if (this.onDrag) {
      this.onDrag({ active: true, startX: this.startNode.centerX, startY: this.startNode.centerY, currX: x, currY: y })
    }
  }

  onUp() {
    this.isDragging = false
    this.swipedInCurrentStroke = false
    if (this.onDrag) this.onDrag({ active: false, startX: 0, startY: 0, currX: 0, currY: 0 })
  }

  onWheel(e) {
    e.preventDefault()
    if (this.swipedInCurrentStroke) return
    this.accumX += e.deltaX
    this.accumY += e.deltaY
    if (Math.abs(this.accumX) > this.wheelThreshold || Math.abs(this.accumY) > this.wheelThreshold) {
      this.swipedInCurrentStroke = true
      this.processWheelGesture()
      clearTimeout(this.wheelResetTimer)
      this.wheelResetTimer = setTimeout(() => {
        this.swipedInCurrentStroke = false
        this.accumX = 0; this.accumY = 0
      }, 1000)
    }
  }

  processWheelGesture() {
    const tx = 20
    const r = this.accumX > tx, l = this.accumX < -tx
    const d = this.accumY > tx, u = this.accumY < -tx
    let type = 'unknown'
    if (r && !u && !d) type = 'right'
    else if (l && !u && !d) type = 'left'
    else if (d && !r && !l) type = 'down'
    else if (u && !r && !l) type = 'up'
    else if (r && d) type = 'diag-dr'
    else if (l && u) type = 'diag-ul'
    else if (l && d) type = 'diag-dl'
    else if (r && u) type = 'diag-ur'
    if (type !== 'unknown' && this.onSwipe) this.onSwipe(type)
  }

  resolveVector(dx, dy) {
    if (dx > 0 && dy === 0) return 'right'
    if (dx < 0 && dy === 0) return 'left'
    if (dx === 0 && dy > 0) return 'down'
    if (dx === 0 && dy < 0) return 'up'
    if (dx > 0 && dy > 0) return 'diag-dr'
    if (dx < 0 && dy < 0) return 'diag-ul'
    if (dx < 0 && dy > 0) return 'diag-dl'
    if (dx > 0 && dy < 0) return 'diag-ur'
    return 'unknown'
  }
}