export function initArticleVisuals() {
  const root = document.querySelector<HTMLElement>('[data-article-visuals]');
  const dialog = document.querySelector<HTMLDialogElement>('[data-image-lightbox]');
  if (!root || !dialog || root.dataset.initialized) return;
  root.dataset.initialized = 'true';
  const article = root.closest('article')!;
  const bodyImages = Array.from(article.querySelectorAll<HTMLImageElement>('.article-content img'));
  const images = bodyImages.length ? bodyImages : Array.from(article.querySelectorAll<HTMLImageElement>('.hero-image img'));
  if (!images.length) return;
  const pick = <T extends Element>(selector: string) => article.querySelector<T>(selector)!;
  const inDialog = <T extends Element>(selector: string) => dialog.querySelector<T>(selector)!;
  const overview = images.findIndex(img => /overview/i.test(img.alt) || /section-overview/.test(img.src));
  if (overview > 0) images.unshift(images.splice(overview, 1)[0]);
  const fullSource = (img: HTMLImageElement) => {
    const candidates = img.srcset.split(',').map(item => item.trim().split(/\s+/));
    const largest = candidates.sort((a, b) => parseFloat(b[1] || '0') - parseFloat(a[1] || '0'))[0];
    return largest?.[0] || img.currentSrc || img.src;
  };
  const headingFor = (img: HTMLImageElement) => {
    const elements = Array.from(article.querySelectorAll('h2, h3, img'));
    const before = elements.slice(0, elements.indexOf(img)).reverse();
    return before.find(el => el.matches('h2, h3'))?.textContent?.trim() || 'Concept overview';
  };
  const slides = images.map(img => ({ image: img, src: fullSource(img), title: headingFor(img), caption: img.alt.trim() || headingFor(img) }));
  const slideImage = pick<HTMLImageElement>('[data-slide-image]');
  const lightImage = inDialog<HTMLImageElement>('[data-lightbox-image]');
  const viewport = inDialog<HTMLElement>('[data-image-viewport]');
  const zoom = inDialog<HTMLButtonElement>('[data-image-zoom]');
  const status = inDialog<HTMLElement>('[data-image-status]');
  let index = 0;
  let returnFocus: HTMLElement | null = null;
  let previousOverflow = '';
  const pageButtons = slides.map((slide, i) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = String(i + 1);
    button.setAttribute('aria-label', `Image ${i + 1}: ${slide.title}`);
    button.addEventListener('click', () => show(i));
    pick('[data-slide-pages]').append(button);
    return button;
  });
  function resetZoom() {
    viewport.classList.remove('is-zoomed');
    viewport.scrollTop = viewport.scrollLeft = 0;
    zoom.textContent = 'Zoom in';
    zoom.setAttribute('aria-label', 'Zoom in');
    zoom.setAttribute('aria-pressed', 'false');
  }
  function show(next: number) {
    index = Math.max(0, Math.min(slides.length - 1, next));
    const slide = slides[index];
    slideImage.src = slide.src;
    slideImage.alt = slide.caption;
    pick('[data-slide-caption]').textContent = slide.caption;
    pick('[data-slide-count]').textContent = `${index + 1} / ${slides.length}`;
    pick<HTMLButtonElement>('[data-slide-prev]').disabled = index === 0;
    pick<HTMLButtonElement>('[data-slide-next]').disabled = index === slides.length - 1;
    pageButtons.forEach((button, i) => button.setAttribute('aria-current', String(i === index)));
    if (dialog!.open) {
      lightImage.src = slide.src;
      lightImage.alt = slide.caption;
      inDialog('[data-image-caption]').textContent = slide.caption;
      inDialog('[data-image-count]').textContent = `${index + 1} / ${slides.length}`;
      inDialog<HTMLButtonElement>('[data-image-prev]').disabled = index === 0;
      inDialog<HTMLButtonElement>('[data-image-next]').disabled = index === slides.length - 1;
      resetZoom();
      status.textContent = copySupported ? '' : 'Copy image is unavailable in this browser. You can download the image instead.';
    }
  }
  function open(i: number, trigger: HTMLElement) {
    returnFocus = trigger;
    previousOverflow = document.body.style.overflow;
    dialog!.showModal();
    document.body.style.overflow = 'hidden';
    show(i);
    inDialog<HTMLButtonElement>('[data-image-close]').focus();
  }
  pick('[data-open-image]').addEventListener('click', () => open(index, pick('[data-open-image]')));
  images.forEach((img, i) => {
    img.classList.add('article-image-trigger');
    img.setAttribute('role', 'button');
    img.tabIndex = 0;
    img.setAttribute('aria-haspopup', 'dialog');
    img.setAttribute('aria-label', `Enlarge image: ${slides[i].caption}`);
    img.addEventListener('click', event => { event.preventDefault(); open(i, img); });
    img.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(i, img); }
    });
  });
  pick('[data-slide-prev]').addEventListener('click', () => show(index - 1));
  pick('[data-slide-next]').addEventListener('click', () => show(index + 1));
  inDialog('[data-image-prev]').addEventListener('click', () => show(index - 1));
  inDialog('[data-image-next]').addEventListener('click', () => show(index + 1));
  inDialog('[data-image-close]').addEventListener('click', () => dialog!.close());
  dialog.addEventListener('close', () => {
    document.body.style.overflow = previousOverflow;
    resetZoom();
    returnFocus?.focus();
  });
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog!.close(); });
  const arrowNavigation = (event: KeyboardEvent) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
    if (dialog!.open && viewport.classList.contains('is-zoomed')) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); show(index + (event.key === 'ArrowRight' ? 1 : -1)); }
  };
  root.addEventListener('keydown', arrowNavigation);
  dialog.addEventListener('keydown', arrowNavigation);
  function swipe(element: HTMLElement, ignore: () => boolean = () => false) {
    let start: { x: number; y: number } | null = null;
    element.addEventListener('touchstart', event => { const t = event.touches[0]; start = t && event.touches.length === 1 ? { x: t.clientX, y: t.clientY } : null; }, { passive: true });
    element.addEventListener('touchend', event => {
      const t = event.changedTouches[0];
      if (start && t && !ignore()) {
        const dx = t.clientX - start.x, dy = t.clientY - start.y;
        if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.5) { show(index + (dx < 0 ? 1 : -1)); suppressClickUntil = Date.now() + 400; }
      }
      start = null;
    }, { passive: true });
  }
  let suppressClickUntil = 0;
  pick('[data-open-image]').addEventListener('click', event => { if (Date.now() < suppressClickUntil) { event.preventDefault(); event.stopImmediatePropagation(); } }, true);
  swipe(pick('[data-open-image]'));
  swipe(viewport, () => viewport.classList.contains('is-zoomed'));
  zoom.setAttribute('aria-pressed', 'false');
  zoom.addEventListener('click', () => {
    const active = viewport.classList.toggle('is-zoomed');
    zoom.textContent = active ? 'Fit image' : 'Zoom in';
    zoom.setAttribute('aria-label', active ? 'Fit image' : 'Zoom in');
    zoom.setAttribute('aria-pressed', String(active));
  });
  async function pngBlob(src: string) {
    const response = await fetch(src);
    if (!response.ok) throw new Error('Image request failed');
    const blob = await response.blob();
    if (blob.type === 'image/png') return blob;
    const url = URL.createObjectURL(blob);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext('2d');
      if (!context) throw new Error('Image conversion unavailable');
      context.drawImage(image, 0, 0);
      return await new Promise<Blob>((resolve, reject) => canvas.toBlob(b => b ? resolve(b) : reject(new Error('Image conversion failed')), 'image/png'));
    } finally { URL.revokeObjectURL(url); }
  }
  const copy = inDialog<HTMLButtonElement>('[data-image-copy]');
  const copySupported = !!navigator.clipboard?.write && typeof ClipboardItem !== 'undefined' && !!window.isSecureContext;
  if (!copySupported) {
    copy.disabled = true;
    copy.title = 'Copy image requires a supported browser on HTTPS or localhost. Use Download instead.';
  }
  copy.addEventListener('click', async () => {
    copy.disabled = true;
    status.textContent = 'Copying image…';
    try {
      // Start the clipboard write during the click; promise-backed data preserves browser user activation.
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': pngBlob(slides[index].src) })]);
      status.textContent = 'Image copied. You can paste it into your notes or slides.';
    } catch { status.textContent = 'Could not copy the image. Use Download, or allow clipboard access in your browser.'; }
    finally { copy.disabled = false; }
  });
  const download = inDialog<HTMLButtonElement>('[data-image-download]');
  download.addEventListener('click', async () => {
    const slide = slides[index];
    const slideNumber = index + 1;
    download.disabled = true;
    status.textContent = 'Preparing download…';
    try {
      const response = await fetch(slide.src);
      if (!response.ok) throw new Error('Image request failed');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      const ext = ({ 'image/png': 'png', 'image/webp': 'webp', 'image/avif': 'avif', 'image/svg+xml': 'svg', 'image/jpeg': 'jpg' } as Record<string, string>)[blob.type.split(';')[0]] || 'png';
      link.href = url;
      link.download = `${location.pathname.split('/').filter(Boolean).pop() || 'article'}-image-${slideNumber}.${ext}`;
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
      status.textContent = 'Download started.';
    } catch { status.textContent = 'Could not download the image. Please try again.'; }
    finally { download.disabled = false; }
  });
  root.hidden = false;
  document.body.classList.add('has-visual-walkthrough');
  show(0);
}
