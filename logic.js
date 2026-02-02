const primitives = ['right', 'left', 'down', 'up', 'diag-dr', 'diag-ul', 'diag-dl', 'diag-ur']
let count = 1
export const MOTIFS = {}
export const SEQUENCE_TIMEOUT = 3000

for (const a of primitives) {
  for (const b of primitives) {
    for (const c of primitives) { MOTIFS[count++] = [a, b, c] }
  }
}

export class SyncBuffer {
  constructor() {
    this.buffer = []
    this.lastTime = 0
  }

  push(type) {
    const now = Date.now()
    if (now - this.lastTime > SEQUENCE_TIMEOUT) this.buffer = []
    this.buffer.push(type)
    this.lastTime = now
    if (this.buffer.length > 3) this.buffer.shift()
    for (const [id, sequence] of Object.entries(MOTIFS)) {
      if (this.buffer.length === 3 && this.buffer.every((v, i) => v === sequence[i])) return parseInt(id)
    }
    return null
  }
}