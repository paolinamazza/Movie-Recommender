// CineMatch AI - Interactive JavaScript
// ======================================

// === STATE MANAGEMENT ===
const state = {
    searching: false,
    currentQuery: '',
    results: null
};

// === DOM ELEMENTS ===
const elements = {
    searchInput: document.getElementById('searchInput'),
    searchBtn: document.getElementById('searchBtn'),
    resultsSection: document.getElementById('results'),
    resultsGrid: document.getElementById('resultsGrid'),
    resultsQuery: document.getElementById('resultsQuery'),
    systemNaive: document.getElementById('systemNaive'),
    systemAdvanced: document.getElementById('systemAdvanced'),
    systemAgentic: document.getElementById('systemAgentic'),
    suggestionChips: document.querySelectorAll('.suggestion-chip'),
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toastMessage')
};

// === EVENT LISTENERS ===
elements.searchBtn.addEventListener('click', handleSearch);
elements.searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSearch();
});

elements.suggestionChips.forEach(chip => {
    chip.addEventListener('click', () => {
        const query = chip.getAttribute('data-query');
        elements.searchInput.value = query;
        handleSearch();
    });
});

// Smooth scroll for nav links
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });

            // Update active nav link
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        }
    });
});

// === SEARCH HANDLER ===
async function handleSearch() {
    const query = elements.searchInput.value.trim();

    if (!query) {
        showToast('⚠️ Please enter a search query', 'warning');
        return;
    }

    if (state.searching) {
        return; // Prevent duplicate searches
    }

    // Get selected systems
    const systems = [];
    if (elements.systemNaive.checked) systems.push('naive');
    if (elements.systemAdvanced.checked) systems.push('advanced');
    if (elements.systemAgentic.checked) systems.push('agentic');

    if (systems.length === 0) {
        showToast('⚠️ Please select at least one RAG system', 'warning');
        return;
    }

    // Update UI state
    state.searching = true;
    state.currentQuery = query;
    elements.searchBtn.classList.add('loading');
    elements.searchBtn.disabled = true;

    try {
        showToast('🔍 Searching across RAG systems...', 'info');

        // Show results section and clear previous results
        elements.resultsQuery.textContent = `"${query}"`;
        elements.resultsSection.style.display = 'block';
        elements.resultsGrid.innerHTML = '';

        // Scroll to results immediately
        setTimeout(() => {
            elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
        }, 100);

        // Create placeholder cards for each system
        const systemConfig = {
            naive: {
                title: 'Naive RAG',
                icon: '<i class="fas fa-medal bronze"></i>',
                badge: 'Bronze'
            },
            advanced: {
                title: 'Advanced RAG',
                icon: '<i class="fas fa-medal silver"></i>',
                badge: 'Silver'
            },
            agentic: {
                title: 'Agentic RAG',
                icon: '<i class="fas fa-medal gold"></i>',
                badge: 'Gold'
            }
        };

        // Add loading placeholders
        systems.forEach(systemKey => {
            const placeholder = createLoadingPlaceholder(systemConfig[systemKey], systemKey);
            elements.resultsGrid.appendChild(placeholder);
        });

        // Make parallel requests for each system
        const promises = systems.map(systemKey =>
            fetchSystemResult(query, systemKey, systemConfig[systemKey])
        );

        // Wait for all to complete
        await Promise.all(promises);

        showToast('✅ All searches completed!', 'success');

    } catch (error) {
        console.error('Search error:', error);
        showToast(`❌ Error: ${error.message}`, 'error');
    } finally {
        state.searching = false;
        elements.searchBtn.classList.remove('loading');
        elements.searchBtn.disabled = false;
    }
}

// === FETCH SINGLE SYSTEM RESULT ===
async function fetchSystemResult(query, systemKey, config) {
    try {
        const response = await fetch(`/api/search/${systemKey}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();

        if (data.success) {
            // Find and replace the loading placeholder
            const placeholder = document.querySelector(`.rag-result.${systemKey}.loading`);
            if (placeholder) {
                const resultCard = createResultCard(data.result, config, systemKey);
                placeholder.replaceWith(resultCard);
            }
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        console.error(`Error fetching ${systemKey}:`, error);
        // Replace placeholder with error message
        const placeholder = document.querySelector(`.rag-result.${systemKey}.loading`);
        if (placeholder) {
            placeholder.innerHTML = `
                <div class="rag-result-header">
                    <div class="rag-result-title">
                        ${config.icon}
                        <span>${config.title}</span>
                        <span class="rag-badge ${systemKey}">${config.badge}</span>
                    </div>
                </div>
                <div style="padding: 2rem; text-align: center; color: var(--secondary);">
                    <i class="fas fa-exclamation-triangle" style="font-size: 2rem; margin-bottom: 1rem;"></i>
                    <div>Error: ${error.message}</div>
                </div>
            `;
            placeholder.classList.remove('loading');
        }
    }
}

// === CREATE LOADING PLACEHOLDER ===
function createLoadingPlaceholder(config, systemKey) {
    const card = document.createElement('div');
    card.className = `rag-result ${systemKey} loading`;

    card.innerHTML = `
        <div class="rag-result-header">
            <div class="rag-result-title">
                ${config.icon}
                <span>${config.title}</span>
                <span class="rag-badge ${systemKey}">${config.badge}</span>
            </div>
        </div>
        <div style="padding: 3rem; text-align: center;">
            <div class="loading-spinner" style="margin: 0 auto 1rem;"></div>
            <div style="color: var(--text-secondary);">Searching...</div>
        </div>
    `;

    return card;
}

// === DISPLAY RESULTS (kept for compatibility) ===
function displayResults(data) {
    elements.resultsQuery.textContent = `"${data.query}"`;
    elements.resultsSection.style.display = 'block';

    // Clear previous results
    elements.resultsGrid.innerHTML = '';

    // Create result cards for each system
    const systemOrder = ['naive', 'advanced', 'agentic'];
    const systemConfig = {
        naive: {
            title: 'Naive RAG',
            icon: '<i class="fas fa-medal bronze"></i>',
            badge: 'Bronze'
        },
        advanced: {
            title: 'Advanced RAG',
            icon: '<i class="fas fa-medal silver"></i>',
            badge: 'Silver'
        },
        agentic: {
            title: 'Agentic RAG',
            icon: '<i class="fas fa-medal gold"></i>',
            badge: 'Gold'
        }
    };

    systemOrder.forEach(systemKey => {
        if (data.results[systemKey]) {
            const result = data.results[systemKey];
            const config = systemConfig[systemKey];

            const resultCard = createResultCard(result, config, systemKey);
            elements.resultsGrid.appendChild(resultCard);
        }
    });
}

// === CREATE RESULT CARD ===
// === CREATE RESULT CARD ===
function createResultCard(result, config, systemKey) {
    const card = document.createElement('div');
    card.className = `rag-result ${systemKey}`;

    // Header
    const header = document.createElement('div');
    header.className = 'rag-result-header';
    header.innerHTML = `
        <div class="rag-result-title">
            ${config.icon}
            <span>${config.title}</span>
            <span class="rag-badge ${systemKey}">${config.badge}</span>
        </div>
    `;
    card.appendChild(header);

    // Performance Metrics
    if (result.metrics) {
        const metrics = document.createElement('div');
        metrics.className = 'rag-result-metrics';

        let metricsHTML = '';

        // Latency
        if (result.metrics.latency_ms !== undefined) {
            metricsHTML += `
                <div class="metric">
                    <span class="metric-value">${result.metrics.latency_ms}ms</span>
                    <span class="metric-label">Latency</span>
                </div>
            `;
        }

        // Cost
        if (result.metrics.est_cost !== undefined) {
            metricsHTML += `
                <div class="metric">
                    <span class="metric-value">$${result.metrics.est_cost}</span>
                    <span class="metric-label">Est. Cost</span>
                </div>
            `;
        }

        // Tokens
        if (result.metrics.est_tokens !== undefined) {
            metricsHTML += `
                <div class="metric">
                    <span class="metric-value">${result.metrics.est_tokens}</span>
                    <span class="metric-label">Tokens</span>
                </div>
            `;
        }

        // Tool calls
        if (result.metrics.tool_calls) {
            metricsHTML += `
                <div class="metric">
                    <span class="metric-value">${result.metrics.tool_calls}</span>
                    <span class="metric-label">Tools Used</span>
                </div>
            `;
        }

        // Reranked
        if (result.metrics.reranked) {
            metricsHTML += `
                <div class="metric">
                    <span class="metric-value">✓</span>
                    <span class="metric-label">Reranked</span>
                </div>
            `;
        }

        // HyDE
        if (result.metrics.hyde_used) {
            metricsHTML += `
                <div class="metric">
                    <span class="metric-value">✓</span>
                    <span class="metric-label">HyDE</span>
                </div>
            `;
        }

        metrics.innerHTML = metricsHTML;
        card.appendChild(metrics);
    }

    // 1. MOVIE GRID (Now comes FIRST)
    if (result.recommendations && result.recommendations.length > 0) {
        const movieGrid = document.createElement('div');
        movieGrid.className = 'movie-grid';
        movieGrid.style.marginBottom = '2rem'; // Add spacing before traces

        result.recommendations.forEach(movie => {
            const movieCard = createMovieCard(movie);
            movieGrid.appendChild(movieCard);
        });

        card.appendChild(movieGrid);
    } else {
        const noResults = document.createElement('div');
        noResults.style.cssText = 'text-align: center; padding: 2rem; color: var(--text-secondary); font-style: italic; margin-bottom: 2rem;';
        noResults.textContent = '🎬 No movies found. Try adjusting your query.';
        card.appendChild(noResults);
    }

    // Helper to create collapsible section
    const createCollapsible = (title, iconClass, contentElement, colorVar) => {
        const container = document.createElement('div');
        container.className = 'collapsible-container';

        const header = document.createElement('div');
        header.className = 'collapsible-header';
        header.innerHTML = `
            <div class="collapsible-title" style="color: ${colorVar || 'var(--text-primary)'}">
                <i class="${iconClass}"></i> ${title}
            </div>
            <i class="fas fa-chevron-down collapsible-icon"></i>
        `;

        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'collapsible-content';
        contentWrapper.appendChild(contentElement);

        header.addEventListener('click', () => {
            header.classList.toggle('active');
            contentWrapper.classList.toggle('open');
        });

        container.appendChild(header);
        container.appendChild(contentWrapper);
        return container;
    };

    // 2. AI REASONING (Collapsible)
    if (result.explanation) {
        const explanationContent = document.createElement('div');
        explanationContent.style.cssText = 'padding: 1rem; background: rgba(255,215,0,0.05); border-left: 3px solid var(--primary); border-radius: 8px; color: var(--text-secondary); line-height: 1.6;';
        explanationContent.innerHTML = result.explanation;

        // Use "AI Explanation" instead of "AI Reasoning" to be more accurate for Naive RAG
        const label = systemKey === 'agentic' ? 'AI Reasoning' : 'AI Explanation';
        card.appendChild(createCollapsible(label, 'fas fa-brain', explanationContent, 'var(--primary)'));
    }

    // 3. PLANNING TRACE (Collapsible - Agentic V2)
    if (result.planning_trace) {
        const planContent = document.createElement('div');
        planContent.style.cssText = 'padding: 1rem; background: rgba(138,43,226,0.1); border-left: 3px solid #8a2be2; border-radius: 8px;';

        const planDetails = document.createElement('div');
        planDetails.style.cssText = 'color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;';
        const stepsCount = result.planning_trace.steps ? result.planning_trace.steps.length : 0;
        planDetails.innerHTML = `
            <div style="margin-bottom: 0.5rem;"><strong>Strategy:</strong> ${result.planning_trace.strategy || 'N/A'}</div>
            <div style="margin-bottom: 0.5rem;"><strong>Steps Planned:</strong> ${stepsCount}</div>
            <div style="margin-bottom: 0.5rem;"><strong>Quality Estimate:</strong> ${result.planning_trace.estimated_quality ? (result.planning_trace.estimated_quality * 100).toFixed(0) + '%' : 'N/A'}</div>
        `;
        planContent.appendChild(planDetails);

        if (result.planning_trace.steps && result.planning_trace.steps.length > 0) {
            const stepsList = document.createElement('div');
            stepsList.style.cssText = 'margin-top: 0.8rem; font-size: 0.85rem;';
            stepsList.innerHTML = '<div style="color: #8a2be2; margin-bottom: 0.5rem; font-weight: 600;">Planned Steps:</div>';

            result.planning_trace.steps.forEach((step, idx) => {
                const stepItem = document.createElement('div');
                stepItem.style.cssText = 'margin-left: 1rem; margin-bottom: 0.3rem; color: var(--text-secondary);';
                stepItem.innerHTML = `${idx + 1}. ${step.tool || 'N/A'} - ${escapeHtml(truncate(step.reasoning || '', 60))}`;
                stepsList.appendChild(stepItem);
            });
            planContent.appendChild(stepsList);
        }

        card.appendChild(createCollapsible('Planning Phase', 'fas fa-map', planContent, '#8a2be2'));
    }

    // 4. EXECUTION TRACE (Collapsible)
    if ((result.tool_trace && result.tool_trace.length > 0) || (result.execution_trace && result.execution_trace.length > 0)) {
        const execContent = document.createElement('div');
        execContent.style.cssText = 'padding: 1rem; background: rgba(0,0,0,0.3); border-radius: 8px; font-family: monospace; font-size: 0.85rem;';

        // Support both V1 and V2 trace formats
        const trace = result.execution_trace || result.tool_trace;

        trace.forEach((step, idx) => {
            const stepDiv = document.createElement('div');
            stepDiv.style.cssText = 'margin: 0.5rem 0; padding: 0.5rem; background: rgba(255,255,255,0.05); border-radius: 4px;';

            // Handle different trace formats
            const toolName = step.tool || 'Unknown Tool';
            const input = step.input || '';
            const output = step.output_summary || step.result_summary || '';
            const success = step.success !== undefined ? step.success : true;

            const statusIcon = success ? '✓' : '✗';
            const statusColor = success ? '#4ade80' : '#f87171';

            stepDiv.innerHTML = `
                <div style="color: var(--primary); display: flex; justify-content: space-between;">
                    <span>Step ${idx + 1}: ${toolName}</span>
                    <span style="color: ${statusColor};">${statusIcon}</span>
                </div>
                <div style="color: var(--text-secondary); font-size: 0.8rem; margin-left: 1rem; margin-top: 0.3rem;">Input: ${escapeHtml(truncate(input, 60))}</div>
                <div style="color: var(--text-secondary); font-size: 0.8rem; margin-left: 1rem;">Output: ${escapeHtml(truncate(output, 80))}</div>
            `;
            execContent.appendChild(stepDiv);
        });

        card.appendChild(createCollapsible('Tool Execution Trace', 'fas fa-cogs', execContent, 'var(--primary)'));
    }

    // 5. REFLECTION (Collapsible - Agentic V2)
    if (result.reflection_results) {
        const reflectContent = document.createElement('div');
        reflectContent.style.cssText = 'padding: 1rem; background: rgba(34,197,94,0.1); border-left: 3px solid #22c55e; border-radius: 8px;';

        const reflectDetails = document.createElement('div');
        reflectDetails.style.cssText = 'color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;';

        let reflectHTML = '';
        if (result.reflection_results.success !== undefined) {
            reflectHTML += `<div style="margin-bottom: 0.5rem;"><strong>Success:</strong> ${result.reflection_results.success ? '✓ Yes' : '✗ No'}</div>`;
        }
        if (result.reflection_results.quality_score !== undefined) {
            const qualityPercent = (result.reflection_results.quality_score * 100).toFixed(0);
            reflectHTML += `<div style="margin-bottom: 0.5rem;"><strong>Quality Score:</strong> ${qualityPercent}%</div>`;
        }
        if (result.reflection_results.issues && result.reflection_results.issues.length > 0) {
            reflectHTML += `<div style="margin-top: 0.8rem;"><strong style="color: #f59e0b;">Issues Found:</strong></div>`;
            result.reflection_results.issues.forEach(issue => {
                reflectHTML += `<div style="margin-left: 1rem; margin-top: 0.3rem; color: #fbbf24;">• ${escapeHtml(issue)}</div>`;
            });
        }
        if (result.reflection_results.suggestions && result.reflection_results.suggestions.length > 0) {
            reflectHTML += `<div style="margin-top: 0.8rem;"><strong style="color: #22c55e;">Suggestions:</strong></div>`;
            result.reflection_results.suggestions.forEach(suggestion => {
                reflectHTML += `<div style="margin-left: 1rem; margin-top: 0.3rem; color: #86efac;">• ${escapeHtml(suggestion)}</div>`;
            });
        }
        reflectDetails.innerHTML = reflectHTML;
        reflectContent.appendChild(reflectDetails);

        card.appendChild(createCollapsible('Reflection & Quality', 'fas fa-lightbulb', reflectContent, '#22c55e'));
    }

    // 6. DETAILED EXPLANATION (Collapsible)
    if (result.detailed_explanation && result.detailed_explanation !== result.explanation) {
        const detailsContent = document.createElement('div');
        detailsContent.style.cssText = 'padding: 1rem; background: rgba(59,130,246,0.1); border-left: 3px solid #3b82f6; border-radius: 8px; color: var(--text-secondary); font-size: 0.9rem; line-height: 1.8; white-space: pre-wrap;';
        detailsContent.textContent = result.detailed_explanation;

        card.appendChild(createCollapsible('Detailed Process', 'fas fa-book-open', detailsContent, '#3b82f6'));
    }

    return card;
}

// === CREATE MOVIE CARD ===
// === CREATE MOVIE CARD ===
function createMovieCard(movie) {
    const container = document.createElement('div');
    container.className = 'movie-card-container';

    const card = document.createElement('div');
    card.className = 'movie-card';
    card.setAttribute('data-movie-id', movie.id || '');

    // FRONT SIDE
    const front = document.createElement('div');
    front.className = 'movie-card-front';

    // Poster with dynamic gradient based on primary genre
    const poster = document.createElement('div');
    poster.className = 'movie-poster';

    // Determine gradient class
    let gradientClass = '';
    const primaryGenre = (movie.genres && movie.genres.length > 0) ? movie.genres[0].toLowerCase() : '';

    if (['action', 'adventure', 'thriller', 'war'].includes(primaryGenre)) {
        gradientClass = 'poster-gradient-action';
    } else if (['comedy', 'family', 'animation', 'romance'].includes(primaryGenre)) {
        gradientClass = 'poster-gradient-comedy';
    } else if (['drama', 'history', 'documentary'].includes(primaryGenre)) {
        gradientClass = 'poster-gradient-drama';
    }
    if (gradientClass) poster.classList.add(gradientClass);

    // Add genre icon
    const icon = document.createElement('i');
    icon.className = 'fas poster-icon';
    if (['action', 'adventure'].includes(primaryGenre)) icon.classList.add('fa-fire');
    else if (['comedy', 'animation'].includes(primaryGenre)) icon.classList.add('fa-laugh-beam');
    else if (['horror', 'thriller'].includes(primaryGenre)) icon.classList.add('fa-ghost');
    else if (['romance', 'family'].includes(primaryGenre)) icon.classList.add('fa-heart');
    else if (['scifi', 'science fiction'].includes(primaryGenre)) icon.classList.add('fa-rocket');
    else icon.classList.add('fa-film'); // Default
    poster.appendChild(icon);

    // Rating badge
    if (movie.tmdb_rating) {
        const rating = document.createElement('div');
        rating.className = 'movie-rating';
        rating.innerHTML = `
            <i class="fas fa-star"></i>
            <span>${movie.tmdb_rating.toFixed(1)}</span>
        `;
        poster.appendChild(rating);
    }

    front.appendChild(poster);

    // Info
    const info = document.createElement('div');
    info.className = 'movie-info';

    const title = document.createElement('div');
    title.className = 'movie-title';
    title.textContent = movie.title || 'Unknown Title';
    title.title = movie.title; // Tooltip for full title
    info.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'movie-meta';
    meta.innerHTML = `
        <span><i class="fas fa-calendar"></i> ${movie.year || 'N/A'}</span>
        <span><i class="fas fa-${movie.type === 'series' ? 'tv' : 'film'}"></i> ${capitalize(movie.type || 'movie')}</span>
    `;
    info.appendChild(meta);

    if (movie.genres && movie.genres.length > 0) {
        const genres = document.createElement('div');
        genres.className = 'movie-genres';
        // Show up to 3 genres, but ensure they wrap properly in CSS
        movie.genres.slice(0, 3).forEach(genre => {
            const tag = document.createElement('span');
            tag.className = 'genre-tag';
            tag.textContent = genre;
            genres.appendChild(tag);
        });
        info.appendChild(genres);
    }

    front.appendChild(info);

    // BACK SIDE
    const back = document.createElement('div');
    back.className = 'movie-card-back';

    const summaryTitle = document.createElement('div');
    summaryTitle.className = 'movie-summary-title';
    summaryTitle.innerHTML = `<i class="fas fa-info-circle"></i> ${movie.title}`;
    back.appendChild(summaryTitle);

    const summaryText = document.createElement('div');
    summaryText.className = 'movie-summary-text';

    // Format duration
    const formatDuration = (runtime) => {
        if (!runtime) return 'N/A';
        const hours = Math.floor(runtime / 60);
        const minutes = runtime % 60;
        if (hours > 0 && minutes > 0) {
            return `${hours}h ${minutes}m`;
        } else if (hours > 0) {
            return `${hours}h`;
        } else {
            return `${minutes}m`;
        }
    };

    // Check if we have a summary
    if (movie.summary || movie.overview || movie.description) {
        // Summary available - show it with metadata
        const metaInfo = document.createElement('div');
        metaInfo.style.cssText = 'margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(255,215,0,0.2); font-size: 0.85rem; color: var(--text-secondary);';

        const metaParts = [];
        if (movie.year) metaParts.push(`📅 ${movie.year}`);
        if (movie.runtime_minutes) metaParts.push(`⏱️ ${formatDuration(movie.runtime_minutes)}`);
        if (movie.tmdb_rating) metaParts.push(`⭐ ${movie.tmdb_rating.toFixed(1)}/10`);

        metaInfo.innerHTML = metaParts.join(' &nbsp;•&nbsp; ');
        summaryText.appendChild(metaInfo);

        const summary = document.createElement('div');
        summary.style.cssText = 'line-height: 1.6;';
        summary.textContent = movie.summary || movie.overview || movie.description;
        summaryText.appendChild(summary);
    } else {
        // No summary - show metadata
        summaryText.innerHTML = `
            <strong>Title:</strong> ${movie.title}<br>
            <strong>Year:</strong> ${movie.year || 'N/A'}<br>
            <strong>Duration:</strong> ${formatDuration(movie.runtime_minutes)}<br>
            <strong>Type:</strong> ${capitalize(movie.type || 'movie')}<br>
            <strong>Rating:</strong> ${movie.tmdb_rating ? movie.tmdb_rating.toFixed(1) : 'N/A'}/10<br>
            ${movie.genres && movie.genres.length > 0 ? `<strong>Genres:</strong> ${movie.genres.join(', ')}` : ''}
        `;
    }
    back.appendChild(summaryText);

    // Assemble card
    card.appendChild(front);
    card.appendChild(back);
    container.appendChild(card);

    // Add flip on click
    container.addEventListener('click', () => {
        card.classList.toggle('flipped');
    });

    return container;
}

// === SHOW MOVIE DETAILS ===
function showMovieDetails(movie) {
    const details = `
        <strong>${movie.title}</strong> (${movie.year})<br>
        ${movie.type === 'series' ? 'Series' : 'Movie'} · ${movie.tmdb_rating ? movie.tmdb_rating.toFixed(1) : 'N/A'}/10<br>
        ${movie.genres ? movie.genres.join(', ') : 'No genres'}
    `;
    showToast(details, 'info', 5000);
}

// === TOAST NOTIFICATION ===
function showToast(message, type = 'info', duration = 3000) {
    elements.toastMessage.innerHTML = message;
    elements.toast.classList.add('show');

    // Auto-hide
    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, duration);
}

// === UTILITY FUNCTIONS ===
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncate(str, maxLength) {
    if (!str) return '';
    return str.length > maxLength ? str.substring(0, maxLength) + '...' : str;
}

function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// === INTERSECTION OBSERVER FOR ANIMATIONS ===
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all sections
document.querySelectorAll('section').forEach(section => {
    section.style.opacity = '0';
    section.style.transform = 'translateY(30px)';
    section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(section);
});

// === LOADING STATE ===
console.log('🎬 CineMatch AI initialized');
console.log('✨ Ready for movie recommendations!');
