// Main script for dynamic content loading, animations, and interactivity

// Global variable to store the current content line canvas
let currentLineCanvas = null;

// Function to show fallback message
function showFallbackMessage(error = "Unknown error") {
    const fallback = document.getElementById('fallback-message');
    fallback.textContent = `Failed to load content: ${error}. Check console (F12).`;
    fallback.style.display = 'block';
    console.error(`Fallback: ${error}`);
}

// Import resume data
import resumeData from './data.js';
console.log('resumeData loaded:', resumeData);

// Function to animate loading sequence
function animateLoadingSequence() {
    const loadingText = document.getElementById('loading-text');
    const accessGranted = document.getElementById('access-granted');
    const overlay = document.getElementById('loading-overlay');
    const lines = [
        "[+] Initializing secure connection...",
        "[+] Connection established.",
        "[+] Decrypting data...",
        "[+] Authorizing.."
    ];

    // Remove orange shadow effect
    loadingText.style.textShadow = 'none';
    loadingText.style.filter = 'none';

    let currentLine = 0;
    let currentText = '';
    let charIndex = 0;

    function type() {
        if (currentLine < lines.length) {
            if (charIndex < lines[currentLine].length) {
                currentText += lines[currentLine][charIndex];
                loadingText.textContent = currentText;
                charIndex++;
                setTimeout(type, 10);
            } else {
                currentText += '\n';
                currentLine++;
                charIndex = 0;
                setTimeout(type, currentLine === 4 ? 1500 : 350);
            }
        } else {
            loadingText.style.display = 'none';
            accessGranted.style.display = 'block';
            setTimeout(() => {
                overlay.style.opacity = '0';
                setTimeout(() => {
                    overlay.style.display = 'none';
                    document.getElementById('intro-section').style.opacity = '1';
                }, 500);
            }, 2000);
        }
    }

    type();
}

// Function to handle photo click
function handlePhotoClick() {
    const photo = document.querySelector('.profile-photo');
    const newPhoto = document.getElementById('new-photo');
    const introSection = document.getElementById('intro-section');
    const nameBlock = document.getElementById('name-block');
    const tickerBlock = document.getElementById('ticker-block');
    const tagsBlock = document.getElementById('tags-block');
    const content = document.getElementById('content');

    // Hide original photo and show new photo
    photo.style.opacity = '0';
    newPhoto.style.display = 'block';
    setTimeout(() => {
        newPhoto.classList.add('visible');
    }, 10);

    // Fade out intro section
    introSection.style.opacity = '0';
    setTimeout(() => {
        introSection.style.display = 'none';
        // Show blocks and content
        nameBlock.style.display = 'block';
        tickerBlock.style.display = 'block';
        tagsBlock.style.display = 'block';
        content.style.display = 'block';
        setTimeout(() => {
            nameBlock.classList.add('visible');
            tickerBlock.classList.add('visible');
            tagsBlock.classList.add('visible');
            content.classList.add('visible');
            // Load Mission Profile by default
            renderSection('about');
        }, 100);
    }, 500);
}

// Function to render section content dynamically
function renderSection(sectionKey) {
    const content = document.getElementById('dynamic-content');
    content.innerHTML = '';
    content.dataset.section = sectionKey; // Store current section
    const section = resumeData[sectionKey];

    if (sectionKey === 'about') {
        const div = document.createElement('div');
        div.innerHTML = `
            <div class="card">
                <p>${section.content}</p>
                <div class="stats">
                    <div><span class="stat-value" data-target="100+">0</span> VAPT Conducted</div>
                    <div><span class="stat-value" data-target="269">0</span> Sectors Secured</div>
                    <div><span class="stat-value" data-target="3">0</span> Years Experience</div>
                </div>
            </div>
        `;
        content.appendChild(div);
        animateStats();
    } else if (sectionKey === 'skills') {
        const div = document.createElement('div');
        div.innerHTML = `
            <div class="skills-filter">
                <button class="filter-btn active" data-category="all">All</button>
                ${section.categories.map(cat => `<button class="filter-btn" data-category="${cat.name}">${cat.name}</button>`).join('')}
            </div>
            <div class="skills-list">
                ${section.categories.map(skill => `
                    <div class="skill-bar" data-category="${skill.name}">
                        <label>${skill.name}</label>
                        <div style="width: ${skill.proficiency}%"></div>
                        <ul class="skill-details" style="display: none;">
                            ${skill.details.map(detail => `<li>${detail}</li>`).join('')}
                        </ul>
                    </div>
                `).join('')}
            </div>
        `;
        content.appendChild(div);
        setupSkillsFilter();
    } else if (sectionKey === 'experience') {
        const div = document.createElement('div');
        div.className = 'timeline';
        div.innerHTML = section.timeline.map(item => `
            <div class="timeline-item card">
                <h3>${item.role}</h3>
                <p>${item.organization} | ${item.period}</p>
                <ul>${item.achievements.map(a => `<li>${a}</li>`).join('')}</ul>
            </div>
        `).join('');
        content.appendChild(div);
    } else if (sectionKey === 'education' || sectionKey === 'certifications') {
        const div = document.createElement('div');
        div.innerHTML = section[sectionKey === 'education' ? 'records' : 'badges'].map(item => `
            <div class="card">
                <h3>${item.degree || item.name}</h3>
                <p>${item.institution || item.issuer} | ${item.period || ''}</p>
                ${item.description ? `<p>${item.description}</p>` : ''}
            </div>
        `).join('');
        content.appendChild(div);
    } else if (sectionKey === 'projects') {
        const div = document.createElement('div');
        div.className = 'project-grid';
        div.innerHTML = section.missions.map((project, index) => `
            <div class="project-card card" data-project-index="${index}">
                <h3>${project.name}</h3>
                <p>${project.description}</p>
            </div>
        `).join('');
        content.appendChild(div);
        setupProjectModals();
    } else if (sectionKey === 'contact') {
        const div = document.createElement('div');
        div.innerHTML = `
            <div class="contact-terminal card">
                <h3>Comm Hub Access</h3>
                <a href="mailto:${section.channels.email}" class="contact-item">Email: ${section.channels.email}</a>
                <a href="${section.channels.linkedin}" target="_blank" class="contact-item">LinkedIn</a>
                <a href="${section.channels.github}" target="_blank" class="contact-item">GitHub</a>
                <a href="${section.channels.medium}" target="_blank" class="contact-item">Medium</a>
            </div>
        `;
        content.appendChild(div);
        setupContactForm();
        animateContactItems();
    }

    // Show content with transition
    const main = document.getElementById('content');
    main.classList.add('visible');
}

// Function to build holographic nodes
function buildHoloNodes() {
    const nodesContainer = document.getElementById('holo-nodes');
    const sections = [
        { id: 'about', label: 'Mission Profile', tooltip: 'View my background' },
        { id: 'skills', label: 'Arsenal', tooltip: 'Explore my skills' },
        { id: 'experience', label: 'Operation Log', tooltip: 'Check my experience' },
        { id: 'education', label: 'Training Academy', tooltip: 'See my education' },
        { id: 'certifications', label: 'Clearance Badges', tooltip: 'View certifications' },
        { id: 'projects', label: 'Mission Archives', tooltip: 'Discover my projects' },
        { id: 'contact', label: 'Comm Hub', tooltip: 'Get in touch' }
    ];

    nodesContainer.innerHTML = sections.map(section => `
        <div class="holo-node" data-section="${section.id}" data-tooltip="${section.tooltip}">${section.label}</div>
    `).join('');

    // Animate nodes
    const nodes = document.querySelectorAll('.holo-node');
    nodes.forEach((node, index) => {
        setTimeout(() => {
            node.classList.add('visible');
        }, index * 200);
    });

    // Setup click handlers
    nodes.forEach(node => {
        node.addEventListener('click', () => {
            console.log(`Holo-node clicked: ${node.dataset.section}`);
            const section = node.dataset.section;
            renderSection(section);
            animateContentLine(node);
        });
    });
}

// Function to animate stats counters
function animateStats() {
    document.querySelectorAll('.stat-value').forEach(stat => {
        const target = stat.dataset.target;
        let current = 0;
        const increment = target.includes('+') ? 10 : target > 100 ? 10 : 1;
        const interval = setInterval(() => {
            current += increment;
            stat.textContent = target.includes('+') ? `${current}+` : current;
            if (current >= parseInt(target)) {
                stat.textContent = target;
                clearInterval(interval);
            }
        }, 50);
    });
}

// Function to animate contact items
function animateContactItems() {
    const items = document.querySelectorAll('.contact-item');
    items.forEach((item, index) => {
        item.style.opacity = '0';
        setTimeout(() => {
            item.style.transition = 'opacity 0.5s';
            item.style.opacity = '1';
        }, index * 300);
    });
}

// Function to handle skills filter
function setupSkillsFilter() {
    const buttons = document.querySelectorAll('.filter-btn');
    console.log(`Found ${buttons.length} filter buttons`);
    buttons.forEach(btn => {
        btn.removeEventListener('click', handleFilterClick); // Prevent duplicate listeners
        btn.addEventListener('click', handleFilterClick);
    });

    function handleFilterClick() {
        console.log(`Filter button clicked: ${this.dataset.category}`);
        buttons.forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        const category = this.dataset.category;
        document.querySelectorAll('.skill-bar').forEach(skill => {
            const details = skill.querySelector('.skill-details');
            if (category === 'all') {
                skill.style.display = 'block';
                details.style.display = 'none';
            } else if (skill.dataset.category === category) {
                skill.style.display = 'block';
                details.style.display = 'block';
            } else {
                skill.style.display = 'none';
                details.style.display = 'none';
            }
        });
    }
}

// Function to handle project modals
function setupProjectModals() {
    const cards = document.querySelectorAll('.project-card');
    const modal = document.getElementById('project-modal');
    const closeBtn = document.getElementById('modal-close');

    console.log(`Found ${cards.length} project cards`);
    cards.forEach(card => {
        card.removeEventListener('click', handleProjectClick); // Prevent duplicate listeners
        card.addEventListener('click', handleProjectClick);
    });

    function handleProjectClick() {
        console.log(`Project card clicked: ${this.dataset.projectIndex}`);
        const index = this.dataset.projectIndex;
        const project = resumeData.projects.missions[index];
        document.getElementById('modal-title').textContent = project.name;
        document.getElementById('modal-description').textContent = project.description;
        const modalLink = document.getElementById('modal-link');
        if (project.link) {
            modalLink.href = project.link;
            modalLink.style.display = 'inline-block';
        } else {
            modalLink.style.display = 'none';
        }
        modal.style.display = 'flex';
    }

    closeBtn.addEventListener('click', () => {
        console.log('Project modal closed');
        modal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            console.log('Project modal closed via backdrop');
            modal.style.display = 'none';
        }
    });
}

// Function to handle contact form modal
function setupContactForm() {
    const modal = document.getElementById('contact-modal');
    const openBtn = document.getElementById('contact-form-btn');
    const closeBtn = document.getElementById('contact-close');
    const form = document.getElementById('contact-form');

    if (!openBtn) {
        console.error('Contact form button not found');
        return;
    }

    openBtn.removeEventListener('click', handleContactOpen); // Prevent duplicate listeners
    openBtn.addEventListener('click', handleContactOpen);

    function handleContactOpen() {
        console.log('Contact form button clicked');
        modal.style.display = 'flex';
    }

    closeBtn.addEventListener('click', () => {
        console.log('Contact modal closed');
        modal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            console.log('Contact modal closed via backdrop');
            modal.style.display = 'none';
        }
    });

    const submitBtn = form.querySelector('#contact-submit');
    submitBtn.removeEventListener('click', handleContactSubmit); // Prevent duplicate listeners
    submitBtn.addEventListener('click', handleContactSubmit);

    function handleContactSubmit() {
        console.log('Contact form submit clicked');
        const name = form.querySelector('#contact-name').value;
        const email = form.querySelector('#contact-email').value;
        const message = form.querySelector('#contact-message').value;
        if (name && email && message) {
            alert('Message received! (Note: This is a demo, no backend is implemented.)');
            modal.style.display = 'none';
            form.reset();
        } else {
            alert('Please fill all fields.');
        }
    }
}

// Function to animate line from node to content
function animateContentLine(node) {
    // Clear previous canvas if exists
    if (currentLineCanvas) {
        currentLineCanvas.remove();
        currentLineCanvas = null;
    }

    // Create new canvas
    const canvas = document.createElement('canvas');
    canvas.className = 'content-line';
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const nodeRect = node.getBoundingClientRect();
    const contentRect = document.getElementById('content').getBoundingClientRect();
    const section = node.dataset.section;

    // Define waypoints for angular path
    const waypoints = [
        { x: nodeRect.right, y: nodeRect.top + nodeRect.height / 2 }, // Start: Right edge of holo-node
        { x: nodeRect.right + 50, y: nodeRect.top + nodeRect.height / 2 }, // Move right
        { x: nodeRect.right + 50, y: contentRect.top + contentRect.height / 2 + 50 }, // Move down
        { x: contentRect.left - 50, y: contentRect.top + contentRect.height / 2 + 50 }, // Move left
        { x: contentRect.left - 50, y: contentRect.top + contentRect.height / 2 }, // Move up
        { x: contentRect.left, y: contentRect.top + contentRect.height / 2 } // End: Left edge of content
    ];

    // Define single label for the horizontal approach
    const labels = [
        { text: section.toUpperCase(), segment: 3, offset: 0.5 } // Section name on horizontal approach
    ];

    let currentSegment = 0;
    let progress = 0;

    function drawLine() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#00e6b8';
        ctx.lineWidth = 2;
        ctx.font = '0.8em Inconsolata';
        ctx.fillStyle = '#ff9444';

        // Draw completed segments
        for (let i = 0; i < currentSegment; i++) {
            ctx.beginPath();
            ctx.moveTo(waypoints[i].x, waypoints[i].y);
            ctx.lineTo(waypoints[i + 1].x, waypoints[i + 1].y);
            ctx.stroke();

            // Draw label for this segment if exists
            const label = labels.find(l => l.segment === i);
            if (label) {
                const lx = waypoints[i].x + (waypoints[i + 1].x - waypoints[i].x) * label.offset;
                const ly = waypoints[i].y + (waypoints[i + 1].y - waypoints[i].y) * label.offset + 10; // Offset below line
                ctx.fillText(label.text, lx, ly);
            }
        }

        // Draw current segment
        if (currentSegment < waypoints.length - 1) {
            const startX = waypoints[currentSegment].x;
            const startY = currentSegment === 0 ? waypoints[currentSegment].y : waypoints[currentSegment].y;
            const endX = waypoints[currentSegment + 1].x;
            const endY = waypoints[currentSegment + 1].y;
            const dx = endX - startX;
            const dy = endY - startY;
            const currentX = startX + dx * progress;
            const currentY = startY + dy * progress;

            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.lineTo(currentX, currentY);
            ctx.stroke();

            // Draw label for current segment if exists
            const label = labels.find(l => l.segment === currentSegment);
            if (label && progress > label.offset) {
                const lx = startX + (endX - startX) * label.offset;
                const ly = startY + (endY - startY) * label.offset + 10;
                ctx.fillText(label.text, lx, ly);
            }

            progress += 0.1; // Increased from 0.05 for faster animation
            if (progress >= 1) {
                progress = 0;
                currentSegment++;
            }
            requestAnimationFrame(drawLine);
        } else {
            // Draw final path
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (let i = 0; i < waypoints.length - 1; i++) {
                ctx.beginPath();
                ctx.moveTo(waypoints[i].x, waypoints[i].y);
                ctx.lineTo(waypoints[i + 1].x, waypoints[i + 1].y);
                ctx.stroke();

                // Draw label
                const label = labels.find(l => l.segment === i);
                if (label) {
                    const lx = waypoints[i].x + (waypoints[i + 1].x - waypoints[i].x) * label.offset;
                    const ly = waypoints[i].y + (waypoints[i + 1].y - waypoints[i].y) * label.offset + 10;
                    ctx.fillText(label.text, lx, ly);
                }
            }
            // Store canvas to keep line visible
            currentLineCanvas = canvas;
        }
    }

    drawLine();
}

// Function to animate CTOS/DedSec HUD with particles, data streams, and flowing lines
function animateHUD() {
    const canvas = document.getElementById('ctos-hud');
    const ctx = canvas.getContext('2d');

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    let scanLineY = 0;
    const particles = Array.from({ length: 50 }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2
    }));

    const dataStreams = Array.from({ length: 8 }, () => ({
        x: Math.random() * canvas.width,
        y: 0,
        text: '10110' + Math.random().toString(2).slice(2, 7),
        speed: 2 + Math.random() * 3
    }));

    const flowLines = Array.from({ length: 5 }, () => ({
        x1: Math.random() * canvas.width,
        y1: Math.random() * canvas.height,
        x2: Math.random() * canvas.width,
        y2: Math.random() * canvas.height,
        vx1: (Math.random() - 0.5) * 1,
        vy1: (Math.random() - 0.5) * 1,
        vx2: (Math.random() - 0.5) * 1,
        vy2: (Math.random() - 0.5) * 1
    }));

    function drawHUD() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#00e6b8';
        ctx.fillStyle = '#00e6b8';
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.3;

        // Scanning line
        ctx.beginPath();
        ctx.moveTo(0, scanLineY);
        ctx.lineTo(canvas.width, scanLineY);
        ctx.stroke();
        scanLineY = (scanLineY + 0.5) % canvas.height;

        // Particles
        ctx.globalAlpha = 0.6;
        particles.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
            ctx.stroke();
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
        });

        // Data streams
        ctx.globalAlpha = 0.5;
        ctx.font = '14px Inconsolata';
        dataStreams.forEach(stream => {
            ctx.fillText(stream.text, stream.x, stream.y);
            stream.y += stream.speed;
            if (stream.y > canvas.height) {
                stream.y = 0;
                stream.x = Math.random() * canvas.width;
                stream.text = '10110' + Math.random().toString(2).slice(2, 7);
            }
        });

        // Flowing lines
        ctx.globalAlpha = 0.4;
        flowLines.forEach(line => {
            ctx.beginPath();
            ctx.moveTo(line.x1, line.y1);
            ctx.lineTo(line.x2, line.y2);
            ctx.stroke();
            line.x1 += line.vx1;
            line.y1 += line.vy1;
            line.x2 += line.vx2;
            line.y2 += line.vy2;
            if (line.x1 < 0 || line.x1 > canvas.width) line.vx1 *= -1;
            if (line.y1 < 0 || line.y1 > canvas.height) line.vy1 *= -1;
            if (line.x2 < 0 || line.x2 > canvas.width) line.vx2 *= -1;
            if (line.y2 < 0 || line.y2 > canvas.height) line.vy2 *= -1;
        });

        requestAnimationFrame(drawHUD);
    }

    drawHUD();
}

// Initialize portfolio
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing portfolio');
    try {
        animateLoadingSequence();
        document.getElementById('profile-img').addEventListener('click', handlePhotoClick);
        buildHoloNodes();
        animateHUD();
    } catch (error) {
        console.error('Error initializing:', error);
        showFallbackMessage(error.message);
    }

    // Hover effects
    document.querySelectorAll('.dedsec-text').forEach(element => {
        element.addEventListener('mouseenter', () => {
            element.style.animation = 'neon-flicker 0.5s';
            setTimeout(() => element.style.animation = 'neon-flicker 2s infinite', 500);
        });
    });

    // Update canvas on resize
    window.addEventListener('resize', () => {
        const hudCanvas = document.getElementById('ctos-hud');
        hudCanvas.width = window.innerWidth;
        hudCanvas.height = window.innerHeight;
        // Redraw current line if exists
        if (currentLineCanvas) {
            const ctx = currentLineCanvas.getContext('2d');
            currentLineCanvas.width = window.innerWidth;
            currentLineCanvas.height = window.innerHeight;
            const node = document.querySelector(`.holo-node[data-section="${document.querySelector('#dynamic-content').dataset.section || 'about'}"]`);
            if (node) {
                const nodeRect = node.getBoundingClientRect();
                const contentRect = document.getElementById('content').getBoundingClientRect();
                ctx.strokeStyle = '#00e6b8';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(nodeRect.left + nodeRect.width / 2, nodeRect.top + nodeRect.height / 2);
                ctx.lineTo(contentRect.left, contentRect.top + contentRect.height / 2);
                ctx.stroke();
            }
        }
    });
});
