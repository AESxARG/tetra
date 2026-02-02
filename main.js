import { mat4 } from 'https://esm.sh/wgpu-matrix'
import { Interface } from './interface.js'
import { SyncBuffer } from './logic.js'

const CONFIG = {
  motifTTL: 3000,
  particleCount: 6000,
  nodeCount: 12,
  rimPower: 7.0,
  rimStrength: 0.6,
  iridescence: 0.01,
  reflection: 0.4,
  explosionForce: 0.25,
  drag: 0.9,
  decay: 0.01,
  camDistance: 60.0,
  camSpeed: 0.0025,
  shadowStrength: 7.5,
  lightPos: [0, 0, 0],
  boundsBase: 5.0,
  speedBase: 0.55,
  turnSpeedBase: 0.35,
  particleSizeBase: 5.0,
  bgAlphaBase: 0.05,
  motif: {
    timeDilation: 0.01,
    decayRate: 0.005,
    flashDecay: 0.9,
    iridescenceBoost: 0.6,
    turnSpeed: 0.95,
    speed: 0.9,
    bounds: 20.0,
    particleSize: 2,
    camSpeed: -0.04
  }
}
const PROFILES = {}, MOTIF_COUNT = 512
const mix = (a, b, t) => a * (1 - t) + b * t
const mixVec = (a, b, t) => a.map((v, i) => v * (1 - t) + b[i] * t)

for (let i = 1; i <= MOTIF_COUNT; i++) {
  const h = (i * (360 / MOTIF_COUNT)) % 360 
  PROFILES[i] = {
    start: hueToRgb(h, 0.9, 0.4), 
    end: hueToRgb((h + 90) % 360, 0.8, 0.3),
    bounds: CONFIG.motif.bounds + (i % 10),
    speed: CONFIG.motif.speed + (i % 5) * 0.02,
    turnSpeed: mix(0.3, CONFIG.motif.turnSpeed, i / MOTIF_COUNT),
    size: CONFIG.motif.particleSize + (i % 3),
    camSpeed: CONFIG.motif.camSpeed
  }
}

const shader = () => `
  struct Uniforms {
    viewProj : mat4x4<f32>,
    lightViewProj : mat4x4<f32>,
    camPos : vec4<f32>,
    colorStart : vec4<f32>,
    colorEnd : vec4<f32>,
    params1 : vec4<f32>,
    params2 : vec4<f32>,
    uiState : vec4<f32>,
    dragStart : vec2<f32>,
    dragEnd : vec2<f32>,
    res : vec2<f32>,
    pad : vec2<f32>
  };
  struct Particle { position: vec4<f32>, velocity: vec4<f32>, color: vec4<f32>, life: f32, scale: f32, seed: f32, pad: f32 };
  struct Node { position: vec4<f32>, velocity: vec4<f32>, targetPos: vec4<f32> };
 
  @group(0) @binding(0) var<uniform> uniforms : Uniforms;
  @group(0) @binding(1) var<storage, read_write> pIO : array<Particle>;
  @group(0) @binding(2) var<storage, read_write> nodes : array<Node>;
  @group(0) @binding(3) var<storage, read> pRead : array<Particle>;
  @group(0) @binding(6) var screenTex: texture_2d<f32>;
  @group(0) @binding(7) var screenSampler: sampler;

  fn rand(co: vec2<f32>) -> f32 { return fract(sin(dot(co, vec2<f32>(12.9898, 78.233))) * 43758.5453); }
  fn spectral(t: f32) -> vec3<f32> { return vec3<f32>(0.5) + vec3<f32>(0.5) * cos(6.28 * (vec3<f32>(1.0) * t + vec3<f32>(0.0, 0.33, 0.67))); }

  @compute @workgroup_size(64)
  fn simulate(@builtin(global_invocation_id) gId : vec3<u32>) {
    let i = gId.x; if (i >= arrayLength(&pIO)) { return; }
    let time = uniforms.params1.x; var p = pIO[i];
    if (i < arrayLength(&nodes)) {
      var n = nodes[i]; let dir = normalize(n.targetPos.xyz - n.position.xyz);
      n.velocity = vec4<f32>(mix(n.velocity.xyz, dir * uniforms.params1.z, uniforms.params1.w), 0.0);
      n.position = n.position + n.velocity;
      if (distance(n.position.xyz, n.targetPos.xyz) < uniforms.params1.y * 0.01) {
        n.targetPos = vec4<f32>((rand(vec2<f32>(time, f32(i))) - 0.5) * uniforms.params1.y * 2.0, (rand(vec2<f32>(time + 1.0, f32(i))) - 0.5) * uniforms.params1.y * 2.0, (rand(vec2<f32>(time + 2.0, f32(i))) - 0.5) * uniforms.params1.y * 1.5, 1.0);
      }
      nodes[i] = n;
    }
    if (p.life <= 0.0) {
      p.position = nodes[u32(rand(vec2<f32>(p.seed, time)) * f32(arrayLength(&nodes))) % arrayLength(&nodes)].position;
      p.velocity = vec4<f32>(cos(rand(vec2<f32>(p.seed, time)) * 6.28), sin(rand(vec2<f32>(p.seed, time)) * 6.28), (rand(vec2<f32>(p.seed + 1.0, time)) - 0.5) * 2.0, 0.0) * ${CONFIG.explosionForce};
      p.life = 1.0; p.scale = 0.0; p.color = vec4<f32>(mix(uniforms.colorStart.rgb, uniforms.colorEnd.rgb, rand(vec2<f32>(p.seed + time, p.seed))), 1.0);
    } else {
      p.position = p.position + p.velocity; p.velocity = p.velocity * ${CONFIG.drag};
      p.life = p.life - ${CONFIG.decay}; p.scale = smoothstep(0.0, 0.2, p.life);
    }
    pIO[i] = p;
  }

  struct VOut { @builtin(position) pos : vec4<f32>, @location(0) color : vec4<f32>, @location(1) norm : vec3<f32>, @location(2) wPos : vec3<f32> };

  @vertex
  fn vs_main(@builtin(instance_index) insIdx : u32, @location(0) position : vec3<f32>, @location(1) normal : vec3<f32>) -> VOut {
    let p = pRead[insIdx];
    let angle = uniforms.params1.x * 5.0 + p.seed * 10.0;
    let c = cos(angle); let s = sin(angle);
    var rPos = vec3<f32>(position.x * c - position.z * s, position.y, position.x * s + position.z * c) * (0.2 + p.scale * 0.5) * uniforms.params2.x;
    var out : VOut; out.pos = uniforms.viewProj * vec4<f32>(rPos + p.position.xyz, 1.0);
    out.wPos = rPos + p.position.xyz; out.norm = normal; out.color = p.color;
    return out;
  }

  @vertex fn vs_quad(@builtin(vertex_index) vIdx: u32) -> @builtin(position) vec4<f32> { var pos = array<vec2<f32>, 3>(vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3)); return vec4<f32>(pos[vIdx], 0, 1); }

  @fragment
  fn fs_main(in : VOut) -> @location(0) vec4<f32> {
    let V = normalize(uniforms.camPos.xyz - in.wPos); let vA = dot(normalize(in.norm), V);
    let fres = pow(1.0 - max(vA, 0.0), ${CONFIG.rimPower});
    let outColor = (in.color.rgb * 0.7) + (mix(vec3<f32>(0.05, 0.05, 0.1), vec3<f32>(0.9), smoothstep(-0.5, 0.8, reflect(-V, normalize(in.norm)).y)) * ${CONFIG.reflection}) + (spectral(vA * 3.0 + uniforms.params1.x * 0.2) * fres * uniforms.params2.z * 1.5);
    return vec4<f32>(outColor, 0.05 + (fres * 0.5));
  }

  @fragment fn fs_fade() -> @location(0) vec4<f32> { 
    return vec4<f32>(mix(vec3<f32>(0.02), uniforms.colorStart.rgb, uniforms.params2.w), uniforms.params2.y + uniforms.params2.w * 0.3); 
  }
  
  @fragment fn fs_ui(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let aspect = uniforms.uiState.z; let dragging = uniforms.uiState.x; let uv = pos.xy / uniforms.res;
    var color = vec4<f32>(0.0);
    for(var i=0; i<3; i++) { for(var j=0; j<3; j++) {
      let nP = vec2<f32>((f32(i) * 2.0 + 1.0) / 6.0, (f32(j) * 2.0 + 1.0) / 6.0);
      var alpha = smoothstep(0.013, 0.008, distance(uv * vec2<f32>(aspect, 1.0), nP * vec2<f32>(aspect, 1.0))) * 0.3;
      if (dragging > 0.5 && distance(uniforms.dragStart * vec2<f32>(aspect, 1.0), nP * vec2<f32>(aspect, 1.0)) < 0.02) { alpha += smoothstep(0.04, 0.0, distance(uv * vec2<f32>(aspect, 1.0), nP * vec2<f32>(aspect, 1.0))) * 0.5; }
      color = max(color, vec4<f32>(1.0, 1.0, 1.0, alpha));
    }}
    if (dragging > ${CONFIG.drag}) {
      let p1 = uniforms.dragStart * vec2<f32>(aspect, 1.0); let p2 = uniforms.dragEnd * vec2<f32>(aspect, 1.0); let ba = p2 - p1;
      let d = length((uv * vec2<f32>(aspect, 1.0) - p1) - ba * clamp(dot(uv * vec2<f32>(aspect, 1.0) - p1, ba) / dot(ba, ba), 0.0, 1.0));
      color = max(color, vec4<f32>(uniforms.colorStart.rgb, smoothstep(0.005, 0.002, d) * 0.8));
    }
    return color;
  }

  @fragment fn fs_screen(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / vec2<f32>(textureDimensions(screenTex)); let d = distance(uv, vec2<f32>(0.5));
    return vec4<f32>(vec3<f32>(textureSample(screenTex, screenSampler, uv - vec2<f32>(d * 0.03, 0.0)).r, textureSample(screenTex, screenSampler, uv).g, textureSample(screenTex, screenSampler, uv + vec2<f32>(d * 0.03, 0.0)).b) * (1.0 - smoothstep(0.5, 1.5, d)), 1.0);
  }
`

function hueToRgb(h, s = 0.9, l = 0.3) {
  const k = n => (n + h / 30) % 12, a = s * Math.min(l, 1 - l)
  const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))
  return [f(0), f(8), f(4)]
}

async function init() {
  if (!navigator.gpu) return
  const adapter = await navigator.gpu.requestAdapter(), device = await adapter.requestDevice(), 
    canvas = document.getElementById('canvas'), syncBuffer = new SyncBuffer()
  const motifContainer = document.getElementById('motif-ui')
  const motifIdDisplay = document.getElementById('motif-id')
  const motifCountDisplay = document.getElementById('motif-count')
  const timerBar = document.getElementById('timer-bar')
  let motifTimer = null
  let dragActive = 0, dragStart = [0, 0], dragCurrent = [0, 0], pulseTime = 0.0, 
    hue1 = 40, hue2 = 50, motifStrength = 0.0, flashStrength = 0.0, activeCategoryId = 0
  const sharedSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
  const getDiscoveredMotifs = () => {
    try {
      const data = localStorage.getItem('tetra_motifs')
      return data ? new Set(JSON.parse(data)) : new Set()
    } catch { return new Set() }
  }
  const saveDiscovery = (id) => {
    const discovered = getDiscoveredMotifs()
    discovered.add(id)
    localStorage.setItem('tetra_motifs', JSON.stringify([...discovered]))
    return discovered.size
  }

  new Interface(canvas, {
    onSwipe: (type) => {
      pulseTime = 1.0; const step = 45
      if (['left', 'right'].includes(type)) hue1 = (hue1 + step) % 360
      else if (['up', 'down'].includes(type)) hue2 = (hue2 + step) % 360
      else if (type.startsWith('diag')) { hue1 = (hue1 + 22.5) % 360; hue2 = (hue2 + 22.5) % 360 }   
      const matchedId = syncBuffer.push(type)
      if (matchedId) { 
        activeCategoryId = matchedId
        motifStrength = 1.0
        flashStrength = 1.0
        const totalFound = saveDiscovery(matchedId)
        motifIdDisplay.innerText = `MOTIF-${matchedId.toString().padStart(3, '0')}`
        motifCountDisplay.innerText = `${totalFound} / ${MOTIF_COUNT} DISCOVERED`
        motifContainer.style.opacity = '1'
        if (motifTimer) clearTimeout(motifTimer)
        motifTimer = setTimeout(() => { motifContainer.style.opacity = '0' }, 8000)
      }
    },
    onDrag: (d) => { if (d.active) { dragActive = 1; dragStart = [d.startX / canvas.width, d.startY / canvas.height]; dragCurrent = [d.currX / canvas.width, d.currY / canvas.height] } else { dragActive = 0 } }
  })

  const context = canvas.getContext('webgpu'), format = navigator.gpu.getPreferredCanvasFormat()
  context.configure({ device, format, alphaMode: 'premultiplied' })
  const vertexBuffer = device.createBuffer({ label: 'Geometry', size: 288, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true })
  new Float32Array(vertexBuffer.getMappedRange()).set([1,1,1,0.57,0.57,0.57,-1,-1,1,0.57,0.57,0.57,-1,1,-1,0.57,0.57,0.57,1,1,1,0.57,-0.57,-0.57,-1,1,-1,0.57,-0.57,-0.57,1,-1,-1,0.57,-0.57,-0.57,1,1,1,-0.57,0.57,-0.57,1,-1,-1,-0.57,0.57,-0.57,-1,-1,1,-0.57,0.57,-0.57,-1,-1,1,-0.57,-0.57,-0.57,1,-1,-1,-0.57,-0.57,-0.57,-1,1,-1,-0.57,-0.57,-0.57]); vertexBuffer.unmap()
  const pBuffer = device.createBuffer({ label: 'Particles', size: CONFIG.particleCount * 64, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true })
  const pArr = new Float32Array(pBuffer.getMappedRange())
  for(let i=0; i<CONFIG.particleCount; i++){ 
    const o=i*16
    pArr[o] = (Math.random() - 0.5) * CONFIG.boundsBase * 2.0 
    pArr[o + 1] = (Math.random() - 0.5) * CONFIG.boundsBase * 2.0
    pArr[o + 2] = (Math.random() - 0.5) * CONFIG.boundsBase
    const c = hueToRgb(Math.random()*360, 0.05, 0.1) 
    pArr[o+8]=c[0]; pArr[o+9]=c[1]; pArr[o+10]=c[2]; pArr[o+11]=1.0
    pArr[o+12]=Math.random(); pArr[o+14]=Math.random() 
  }
  pBuffer.unmap()
  const nBuffer = device.createBuffer({ label: 'Nodes', size: CONFIG.nodeCount * 48, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true })
  const nArr = new Float32Array(nBuffer.getMappedRange()); for(let i=0; i<CONFIG.nodeCount; i++){ const o=i*12; nArr[o]=(Math.random()-0.5)*20; nArr[o+8]=(Math.random()-0.5)*20 }; nBuffer.unmap()
  const uBuffer = device.createBuffer({ label: 'Uniforms', size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), shaderMod = device.createShaderModule({ code: shader() })
  const layout0 = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] })
  const layout2 = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }] })
  const layout3 = device.createBindGroupLayout({ entries: [{ binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }, { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } }] })
  const cBG = device.createBindGroup({ layout: layout0, entries: [{ binding: 0, resource: { buffer: uBuffer } }, { binding: 1, resource: { buffer: pBuffer } }, { binding: 2, resource: { buffer: nBuffer } }] })
  const mBG = device.createBindGroup({ layout: layout2, entries: [{ binding: 0, resource: { buffer: uBuffer } }, { binding: 3, resource: { buffer: pBuffer } }] })
  const pipe = {
    c: device.createComputePipeline({ label: 'Sim', layout: device.createPipelineLayout({ bindGroupLayouts: [layout0] }), compute: { module: shaderMod, entryPoint: 'simulate' } }),
    r: device.createRenderPipeline({ label: 'Main', layout: device.createPipelineLayout({ bindGroupLayouts: [layout2] }), vertex: { module: shaderMod, entryPoint: 'vs_main', buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3'}, { shaderLocation: 1, offset: 12, format: 'float32x3'}] }] }, fragment: { module: shaderMod, entryPoint: 'fs_main', targets: [{ format, blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] }, primitive: { topology: 'triangle-list' }, depthStencil: { depthWriteEnabled: false, depthCompare: 'less', format: 'depth24plus' } }),
    f: device.createRenderPipeline({ label: 'Fade', layout: device.createPipelineLayout({ bindGroupLayouts: [layout2] }), vertex: { module: shaderMod, entryPoint: 'vs_quad' }, fragment: { module: shaderMod, entryPoint: 'fs_fade', targets: [{ format, blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] }, primitive: { topology: 'triangle-list' }, depthStencil: { depthWriteEnabled: false, depthCompare: 'always', format: 'depth24plus' } }),
    u: device.createRenderPipeline({ label: 'UI', layout: device.createPipelineLayout({ bindGroupLayouts: [layout2] }), vertex: { module: shaderMod, entryPoint: 'vs_quad' }, fragment: { module: shaderMod, entryPoint: 'fs_ui', targets: [{ format, blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }] }, primitive: { topology: 'triangle-list' }, depthStencil: { depthWriteEnabled: false, depthCompare: 'always', format: 'depth24plus' } }),
    scr: device.createRenderPipeline({ label: 'Scr', layout: device.createPipelineLayout({ bindGroupLayouts: [layout3] }), vertex: { module: shaderMod, entryPoint: 'vs_quad' }, fragment: { module: shaderMod, entryPoint: 'fs_screen', targets: [{ format }] }, primitive: { topology: 'triangle-list' } })
  }
  let dTex, rTex, rView, scrBG, time = 0; const uData = new Float32Array(64)

  function frame() {
    if (!dTex || !rView || !scrBG) { requestAnimationFrame(frame); return }
    if (motifStrength > 0) motifStrength = Math.max(0, motifStrength - CONFIG.motif.decayRate)
    if (flashStrength > 0) flashStrength = Math.max(0, flashStrength - CONFIG.motif.flashDecay)
    const now = Date.now()
    const timeSinceLast = now - syncBuffer.lastTime
    if (syncBuffer.buffer.length > 0 && timeSinceLast < CONFIG.motifTTL) {
      const remaining = 1.0 - (timeSinceLast / CONFIG.motifTTL)
      timerBar.style.width = `${remaining * 100}%`
      timerBar.style.opacity = '1'
    } else {
      timerBar.style.width = '0%'
      timerBar.style.opacity = '0'
    }
    const prof = PROFILES[activeCategoryId] || { 
      start: [0.15, 0.15, 0.2], 
      end: [0.1, 0.1, 0.15], 
      bounds: CONFIG.boundsBase, 
      speed: CONFIG.speedBase, 
      turnSpeed: CONFIG.turnSpeedBase, 
      size: CONFIG.particleSizeBase, 
      camSpeed: CONFIG.camSpeed 
    }
    const currentCamSpeed = mix(CONFIG.camSpeed, prof.camSpeed, motifStrength)
    const dilation = 1.0 - (motifStrength * (1.0 - CONFIG.motif.timeDilation))
    time += currentCamSpeed * dilation
    const asp = canvas.width / canvas.height
    const vp = mat4.multiply(
      mat4.perspective(Math.PI / 4, asp, 0.1, 200), 
      mat4.lookAt([Math.sin(time * 0.5) * 60, Math.cos(time * 0.2) * 18, Math.cos(time * 0.5) * 60], [0, 0, 0], [0, 1, 0]
    ))
    const lv = mat4.multiply(
      mat4.ortho(-60, 60, -60, 60, 1, 200), mat4.lookAt(CONFIG.lightPos, [0, 0, 0], [0, 1, 0])
    )
    const baseC1 = hueToRgb(hue1, 0.02, 0.01), baseC2 = hueToRgb(hue2, 0.02, 0.01)
    const c1 = mixVec(baseC1, prof.start, motifStrength)
    const c2 = mixVec(baseC2, prof.end, motifStrength)
    const db = mix(CONFIG.boundsBase, prof.bounds, motifStrength)
    const ds = mix(CONFIG.speedBase, prof.speed, motifStrength)
    const dt = mix(CONFIG.turnSpeedBase, prof.turnSpeed, motifStrength)
    const dp = mix(CONFIG.particleSizeBase, prof.size, motifStrength)
    const di = CONFIG.iridescence + (CONFIG.motif.iridescenceBoost * motifStrength)
    uData.set(vp, 0)
    uData.set(lv, 16)
    uData.set([0, 0, 40, 1], 32)
    uData.set([...c1, 1], 36)
    uData.set([...c2, 1], 40)
    uData.set([time, db, ds, dt], 44)
    uData.set([dp, CONFIG.bgAlphaBase, di, flashStrength], 48)
    uData.set([dragActive, pulseTime, asp, 0], 52)
    uData.set(dragStart, 56)
    uData.set(dragCurrent, 58)
    uData.set([canvas.width, canvas.height], 60)
    device.queue.writeBuffer(uBuffer, 0, uData)
    const enc = device.createCommandEncoder()
    const cp = enc.beginComputePass()
    cp.setPipeline(pipe.c); cp.setBindGroup(0, cBG); cp.dispatchWorkgroups(Math.ceil(CONFIG.particleCount / 64)); cp.end()
    const op = enc.beginRenderPass({ 
      colorAttachments: [{ view: rView, loadOp: 'load', storeOp: 'store' }], 
      depthStencilAttachment: { view: dTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } 
    })
    op.setPipeline(pipe.f); op.setBindGroup(0, mBG); op.draw(3)
    op.setPipeline(pipe.r); op.setVertexBuffer(0, vertexBuffer); op.setBindGroup(0, mBG); op.draw(12, CONFIG.particleCount)
    op.setPipeline(pipe.u); op.setBindGroup(0, mBG); op.draw(3); op.end()
    const scp = enc.beginRenderPass({ 
      colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue: {r:0,g:0,b:0,a:1}, loadOp: 'clear', storeOp: 'store' }] 
    })
    scp.setPipeline(pipe.scr); scp.setBindGroup(0, scrBG); scp.draw(3); scp.end()
    device.queue.submit([enc.finish()])
    requestAnimationFrame(frame)
  }

  new ResizeObserver(e => {
    const w = Math.max(1, Math.min(e[0].contentRect.width, device.limits.maxTextureDimension2D))
    const h = Math.max(1, Math.min(e[0].contentRect.height, device.limits.maxTextureDimension2D))
    canvas.width = w; canvas.height = h
    if (dTex) dTex.destroy()
    dTex = device.createTexture({ size: [w, h], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT })
    if (rTex) rTex.destroy()
    rTex = device.createTexture({ size: [w, h], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING })
    rView = rTex.createView()
    scrBG = device.createBindGroup({ 
      layout: layout3, 
      entries: [ { binding: 6, resource: rView }, { binding: 7, resource: sharedSampler } ]
    })
  }).observe(canvas)
  frame()
}
init()