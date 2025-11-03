const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');
const colors = require('colors');

// Configuration
const ASSETS = ['BTC', 'ETH', 'SOL', 'XRP'];
const TIMEFRAMES = ['m15', 'h1', 'daily'];
const DATA_DIR = path.join(__dirname, '..', 'data');

// Flag debug depuis argv
const DEBUG_MODE = process.argv.includes('--debug') || process.argv.includes('-d');

// Couleurs pour chaque asset
const ASSET_COLORS = {
    'BTC': colors.yellow.bold,
    'ETH': colors.blue.bold,
    'SOL': colors.magenta.bold,
    'XRP': colors.cyan.bold
};

// Structure de stockage des marchés
let MARKETS = {};

// Buffers pour les données
const BUFFERS = {};

// Timestamps des bougies actives
const ACTIVE_BOUGIES = {};

// Initialisation
ASSETS.forEach(asset => {
    MARKETS[asset] = { m15: null, h1: null, daily: null };
    BUFFERS[asset] = [];
    ACTIVE_BOUGIES[asset] = { m15: null, h1: null, daily: null };
});

// Créer le dossier data s'il n'existe pas
if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    console.log(`✓ Dossier créé: ${DATA_DIR}`);
}

/**
 * Parse le slug et extrait les informations de la bougie
 */
function parseSlug(slug, asset, timeframeOverride = null) {
    // Générer les patterns possibles selon l'asset
    const fullNames = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'SOL': 'solana',
        'XRP': 'xrp'
    };
    const shortName = asset.toLowerCase();
    const fullName = fullNames[asset] || shortName;
    
    // Nouveau format: btc-updown-15m-1762104600 (timestamp Unix)
    const unixPattern = new RegExp(`(${shortName}|${fullName})-updown-(15m|1h|1d)-(\\d+)`);
    const unixMatch = slug.match(unixPattern);
    
    if (unixMatch) {
        const tfMatch = unixMatch[2];
        const unixTimestamp = parseInt(unixMatch[3]);
        
        let timeframe;
        if (tfMatch === '15m') timeframe = 'm15';
        else if (tfMatch === '1h') timeframe = 'h1';
        else if (tfMatch === '1d') timeframe = 'daily';
        else return null;
        
        // Le timestamp Unix représente le début de la bougie en ET
        // Convertir en Date (JS attend milliseconds)
        const timestamp = new Date(unixTimestamp * 1000);
        
        // Retourner le début de la bougie (pas la fin)
        return { timeframe, timestamp };
    }
    
    // Format daily: bitcoin-up-or-down-on-november-3
    const dailyPattern = new RegExp(`(${shortName}|${fullName})-up-or-down-on-([a-z]+)-(\\d+)`);
    const dailyMatch = slug.match(dailyPattern);
    if (dailyMatch) {
        const monthName = dailyMatch[2];
        const dayNum = parseInt(dailyMatch[3]);

        // Convertir mois anglais -> numéro
        const MONTHS = {
            january: 1, february: 2, march: 3, april: 4, may: 5, june: 6,
            july: 7, august: 8, september: 9, october: 10, november: 11, december: 12
        };
        const monthNum = MONTHS[monthName.toLowerCase()];
        if (!monthNum) return null;

        // Déterminer l'année (si la date est déjà passée cette année en ET, prendre l'année suivante)
        const now = new Date();
        const nowStr = now.toLocaleString('en-US', { timeZone: 'America/New_York' });
        const nowParts = nowStr.match(/(\d+)\/(\d+)\/(\d+), (\d+):(\d+):(\d+)/);
        if (!nowParts) return null;
        let year = parseInt(nowParts[3]);
        
        // Comparer la date candidat avec now (en ET)
        const candidateET = new Date(year, monthNum - 1, dayNum, 0, 0, 0, 0);
        const nowET = new Date(parseInt(nowParts[3]), parseInt(nowParts[1]) - 1, parseInt(nowParts[2]), 
                                parseInt(nowParts[4]), parseInt(nowParts[5]), parseInt(nowParts[6]));
        if (candidateET < nowET) {
            year += 1;
        }

        // Construire une Date pour le début de la journée en ET
        const tzParts = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', timeZoneName: 'short' })
            .formatToParts(new Date(year, monthNum - 1, dayNum, 0, 0, 0, 0));
        const tzName = tzParts.find(p => p.type === 'timeZoneName')?.value || 'EST';
        const offset = tzName === 'EDT' ? '-04:00' : '-05:00';
        const iso = `${year}-${String(monthNum).padStart(2,'0')}-${String(dayNum).padStart(2,'0')}T00:00:00${offset}`;
        const timestamp = new Date(iso);

        return { timeframe: 'daily', timestamp };
    }
    
    // Format h1: xrp-up-or-down-november-4-12pm-et
    const h1Pattern = new RegExp(`(${shortName}|${fullName})-up-or-down-([a-z]+)-(\\d+)-(\\d+)(am|pm)-et`);
    const h1Match = slug.match(h1Pattern);
    if (h1Match) {
        const monthName = h1Match[2];
        const dayNum = parseInt(h1Match[3]);
        let hourNum = parseInt(h1Match[4]);
        const ampm = h1Match[5];

        // Convertir mois anglais -> numéro
        const MONTHS = {
            january: 1, february: 2, march: 3, april: 4, may: 5, june: 6,
            july: 7, august: 8, september: 9, october: 10, november: 11, december: 12
        };
        const monthNum = MONTHS[monthName.toLowerCase()];
        if (!monthNum) return null;

        if (ampm === 'pm' && hourNum < 12) hourNum += 12;
        if (ampm === 'am' && hourNum === 12) hourNum = 0;

        // Déterminer l'année (si la date est déjà passée cette année en ET, prendre l'année suivante)
        const now = new Date();
        const nowStr = now.toLocaleString('en-US', { timeZone: 'America/New_York' });
        const nowParts = nowStr.match(/(\d+)\/(\d+)\/(\d+), (\d+):(\d+):(\d+)/);
        if (!nowParts) return null;
        let year = parseInt(nowParts[3]);
        
        // Comparer la date candidat avec now (en ET)
        const candidateET = new Date(year, monthNum - 1, dayNum, hourNum, 0, 0, 0);
        const nowET = new Date(parseInt(nowParts[3]), parseInt(nowParts[1]) - 1, parseInt(nowParts[2]), 
                                parseInt(nowParts[4]), parseInt(nowParts[5]), parseInt(nowParts[6]));
        if (candidateET < nowET) {
            year += 1;
        }

        // Construire une Date avec offset ET correct (DST)
        // Détecter EST/EDT pour cette date
        const tzParts = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', timeZoneName: 'short' })
            .formatToParts(new Date(year, monthNum - 1, dayNum, hourNum, 0, 0, 0));
        const tzName = tzParts.find(p => p.type === 'timeZoneName')?.value || 'EST';
        const offset = tzName === 'EDT' ? '-04:00' : '-05:00';
        const iso = `${year}-${String(monthNum).padStart(2,'0')}-${String(dayNum).padStart(2,'0')}T${String(hourNum).padStart(2,'0')}:00:00${offset}`;
        const timestamp = new Date(iso);

        return { timeframe: 'h1', timestamp };
    }

    // Ancien format (backup): btc-updown-15m-2025-11-02-13-30
    const patterns = {
        m15: new RegExp(`${asset.toLowerCase()}-updown-15m-(\\d{4})-(\\d{2})-(\\d{2})-(\\d{2})-(\\d{2})`),
        h1: new RegExp(`${asset.toLowerCase()}-updown-1h-(\\d{4})-(\\d{2})-(\\d{2})-(\\d{2})`),
        daily: new RegExp(`${asset.toLowerCase()}-updown-1d-(\\d{4})-(\\d{2})-(\\d{2})`)
    };

    for (const [tf, pattern] of Object.entries(patterns)) {
        const match = slug.match(pattern);
        if (match) {
            const [year, month, day, hour, minute] = match.slice(1).map(x => parseInt(x));
            
            // Créer une date interprétée comme heure ET locale
            let dateStr;
            if (tf === 'm15') {
                dateStr = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}T${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:00`;
            } else if (tf === 'h1') {
                dateStr = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}T${String(hour).padStart(2, '0')}:00:00`;
            } else {
                dateStr = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}T23:59:59`;
            }
            
            // Vérifier si on est en heure d'été (EDT vs EST)
            const testDate = new Date(`${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}T12:00:00Z`);
            const etTest = new Intl.DateTimeFormat('en-US', {
                timeZone: 'America/New_York',
                timeZoneName: 'short'
            }).formatToParts(testDate);
            const isDST = etTest.find(p => p.type === 'timeZoneName').value === 'EDT';
            
            const finalDateStr = isDST ? `${dateStr}-04:00` : `${dateStr}-05:00`;
            const timestamp = new Date(finalDateStr);
            
            // Ajouter la durée de la bougie
            if (tf === 'm15') {
                timestamp.setMinutes(timestamp.getMinutes() + 15);
            } else if (tf === 'h1') {
                timestamp.setHours(timestamp.getHours() + 1);
            }
            
            return { timeframe: tf, timestamp };
        }
    }
    return null;
}

/**
 * Détermine quelle bougie est active pour un timeframe donné
 * Retourne un timestamp en UTC qui représente le début/fin de la bougie
 */
function getActiveBougie(asset, timeframe, now = new Date()) {
    // Convertir maintenant en heures ET
    const nowET = new Intl.DateTimeFormat('en-US', {
        timeZone: 'America/New_York',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
    }).formatToParts(now);
    
    const year = parseInt(nowET.find(p => p.type === 'year').value);
    const month = parseInt(nowET.find(p => p.type === 'month').value) - 1;
    const day = parseInt(nowET.find(p => p.type === 'day').value);
    const hour = parseInt(nowET.find(p => p.type === 'hour').value);
    const minute = parseInt(nowET.find(p => p.type === 'minute').value);
    
    // Calculer la prochaine bougie en ET
    let nextHour = hour;
    let nextMinute = minute;
    
    if (timeframe === 'm15') {
        nextMinute = Math.ceil(minute / 15) * 15;
        if (nextMinute === 60) {
            nextMinute = 0;
            nextHour = hour + 1;
        }
    } else if (timeframe === 'h1') {
        nextHour = hour + 1;
        nextMinute = 0;
    } else if (timeframe === 'daily') {
        // Pour daily, on veut la bougie du jour actuel (minuit du jour)
        nextHour = 0;
        nextMinute = 0;
    }
    
    // Créer une string de date ET et la convertir en Date
    const etDateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')} ${String(nextHour).padStart(2, '0')}:${String(nextMinute).padStart(2, '0')}:00`;
    
    // Utiliser une approche simple: on crée une date en interprétant comme si c'était UTC
    // Puis on ajuste pour avoir l'équivalent ET
    const testDate = new Date(Date.UTC(year, month, day, nextHour, nextMinute, 0, 0));
    
    // Obtenir l'heure ET de cette date de test
    const testETStr = testDate.toLocaleString('en-US', { timeZone: 'America/New_York', hour12: false });
    const testETMatch = testETStr.match(/(\d+)\/(\d+)\/(\d+), (\d+):(\d+):(\d+)/);
    
    if (testETMatch) {
        const [_, testMonth, testDay, testYear, testHour, testMin] = testETMatch;
        
        // Calculer la différence
        const diffHour = nextHour - parseInt(testHour);
        
        // Ajuster
        const targetDate = new Date(Date.UTC(year, month, day, nextHour - diffHour, nextMinute, 0, 0));
        return targetDate;
    }
    
    // Fallback: utiliser EST/EDT approximatif
    const isDST = new Date().getTimezoneOffset() > 300;
    const offset = isDST ? -4 : -5;
    return new Date(Date.UTC(year, month, day, nextHour - offset, nextMinute, 0, 0));
}

/**
 * Génère le slug Polymarket attendu pour une bougie
 */
function generateExpectedSlug(asset, timeframe, now = new Date()) {
    // Obtenir les composantes ET de "now"
    const etFmt = new Intl.DateTimeFormat('en-US', {
        timeZone: 'America/New_York',
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
    });
    
    const nowParts = etFmt.formatToParts(now);
    const get = (parts, type) => parseInt(parts.find(p => p.type === type).value);
    const nowY = get(nowParts, 'year');
    const nowMo = get(nowParts, 'month');
    const nowD = get(nowParts, 'day');
    const nowH = get(nowParts, 'hour');
    const nowMi = get(nowParts, 'minute');

    // Calculer la prochaine bougie
    let targetY = nowY, targetMo = nowMo, targetD = nowD, targetH = nowH, targetMi = nowMi;
    
    if (timeframe === 'm15') {
        targetMi = Math.ceil(nowMi / 15) * 15;
        if (targetMi === 60) {
            targetMi = 0;
            targetH = (nowH + 1) % 24;
            if (targetH === 0) {
                targetD++;
            }
        }
    } else if (timeframe === 'h1') {
        // Pas de décalage : on reste sur l'heure actuelle
        targetH = nowH;
        targetMi = 0;
    } else if (timeframe === 'daily') {
        targetH = 0;
        targetMi = 0;
        // Les paris daily se mettent à jour à 12pm ET
        // Si on est >= 12h ET, c'est le marché du lendemain
        if (nowH >= 12) {
            targetD++;
        }
    }
    
    // Créer le slug selon le format Polymarket
    // m15 utilise les codes courts, h1/daily utilisent les noms complets
    let assetStr;
    if (timeframe === 'm15') {
        assetStr = asset.toLowerCase(); // btc, eth, sol, xrp
    } else {
        const fullNames = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'XRP': 'xrp'
        };
        assetStr = fullNames[asset] || asset.toLowerCase();
    }
    
    if (timeframe === 'm15') {
        // Format: btc-updown-15m-1762120800 (timestamp Unix)
        const unixTs = Math.floor(Date.UTC(targetY, targetMo - 1, targetD, targetH, targetMi, 0) / 1000);
        return `${assetStr}-updown-15m-${unixTs}`;
    } else if (timeframe === 'h1') {
        // Format: bitcoin-up-or-down-november-2-3pm-et
        const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june',
                       'july', 'august', 'september', 'october', 'november', 'december'];
        const monthName = MONTHS[targetMo - 1];
        const hour12 = targetH === 0 ? 12 : (targetH > 12 ? targetH - 12 : targetH);
        const ampm = targetH < 12 ? 'am' : 'pm';
        return `${assetStr}-up-or-down-${monthName}-${targetD}-${hour12}${ampm}-et`;
    } else if (timeframe === 'daily') {
        // Format: bitcoin-up-or-down-on-november-3
        const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june',
                       'july', 'august', 'september', 'october', 'november', 'december'];
        const monthName = MONTHS[targetMo - 1];
        return `${assetStr}-up-or-down-on-${monthName}-${targetD}`;
    }
    
    return null;
}

/**
 * Vérifie si un marché correspond à la bougie active
 * Retourne true si "now" est dans la bougie définie par parsed.timestamp
 */
function isActiveMarket(parsed, asset, timeframe, now = new Date()) {
    if (!parsed) return [false, null];

    // Obtenir les composantes ET de "now" et du marché
    const etFmt = new Intl.DateTimeFormat('en-US', {
        timeZone: 'America/New_York',
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
    });
    
    const nowParts = etFmt.formatToParts(now);
    const mParts = etFmt.formatToParts(parsed.timestamp);
    const get = (parts, type) => parseInt(parts.find(p => p.type === type).value);
    
    const nowY = get(nowParts, 'year');
    const nowMo = get(nowParts, 'month');
    const nowD = get(nowParts, 'day');
    const nowH = get(nowParts, 'hour');
    const nowMi = get(nowParts, 'minute');
    
    const mY = get(mParts, 'year');
    const mMo = get(mParts, 'month');
    const mD = get(mParts, 'day');
    const mH = get(mParts, 'hour');
    const mMi = get(mParts, 'minute');

    // Comparer si "now" est dans la même bougie que le marché
    let matches = false;
    
    if (timeframe === 'm15') {
        // Pour m15, vérifier que now est dans les 15 minutes de la bougie
        // Ex: bougie 14:00 -> valide de 14:00 à 14:14
        matches = nowY === mY && nowMo === mMo && nowD === mD && nowH === mH && 
                  nowMi >= mMi && nowMi < mMi + 15;
    } else if (timeframe === 'h1') {
        // Pour h1, vérifier que now est dans la même heure
        // Ex: bougie 14:00 -> valide de 14:00 à 14:59
        matches = nowY === mY && nowMo === mMo && nowD === mD && nowH === mH;
    } else if (timeframe === 'daily') {
        // Pour daily, vérifier si le marché correspond
        // Les paris daily se mettent à jour à 12pm ET
        // Si now >= 12h ET, on vérifie le lendemain
        let expectedD = nowD;
        if (nowH >= 12) {
            expectedD = nowD + 1;
            // Gérer fin de mois
            const daysInMonth = new Date(nowY, nowMo, 0).getDate();
            if (expectedD > daysInMonth) {
                expectedD = 1;
                // Gérer année suivante si besoin
            }
        }
        matches = nowY === mY && nowMo === mMo && expectedD === mD;
    }
    
    return [matches, parsed.timestamp];
}

/**
 * Récupère les marchés Polymarket et met à jour les clobTokenIds
 */
async function refreshMarkets() {
    
    try {
        const response = await fetch('https://gamma-api.polymarket.com/events?closed=false&limit=100&order=id&ascending=false');
        const data = await response.json();
        
        const markets = data.events || data;
        
        if (!markets || !Array.isArray(markets)) {
            console.log('✗ Réponse invalide de l\'API Gamma');
            return;
        }


        const now = new Date();
        const nowET = new Date().toLocaleString('en-US', { timeZone: 'America/New_York' });
        
        const newMarkets = {};

        ASSETS.forEach(asset => {
            newMarkets[asset] = { m15: null, h1: null, daily: null };
        });

        let parsedCount = 0;
        let activeCount = 0;
        let clobCount = 0;

        for (const event of markets) {
            const slug = event?.slug;
            if (!slug) continue;

            // Déterminer l'asset (accepte formats updown & up-or-down)
            let asset;
            if (slug.startsWith('btc-') || slug.includes('bitcoin-')) asset = 'BTC';
            else if (slug.startsWith('eth-') || slug.includes('ethereum-')) asset = 'ETH';
            else if (slug.startsWith('sol-') || slug.includes('solana-')) asset = 'SOL';
            else if (slug.startsWith('xrp-')) asset = 'XRP';
            else continue;
            
            parsedCount++;

            // Déterminer timeframe via série si disponible
            let tfHint = null;
            try {
                const rec = Array.isArray(event.series) && event.series.length > 0 ? (event.series[0]?.recurrence || null) : null;
                if (rec) {
                    const r = String(rec).toLowerCase();
                    if (r.includes('15')) tfHint = 'm15';
                    else if (r.includes('1h') || r.includes('hour')) tfHint = 'h1';
                    else if (r.includes('day') || r.includes('daily') || r.includes('1d') || r.includes('24')) tfHint = 'daily';
                }
            } catch (_) {}

            const parsed = parseSlug(slug, asset, tfHint);
            if (!parsed) continue;
            
            // Vérifier que c'est la bougie active (prochaine)
            const [isActive] = isActiveMarket(parsed, asset, parsed.timeframe, now);
            
            if (isActive) {
                activeCount++;
                // Extraire clobTokenIds depuis l'objet interne event.markets
                let clobs = null;
                if (Array.isArray(event.markets) && event.markets.length > 0) {
                    // Prendre le premier market qui a des clobTokenIds
                    for (const inner of event.markets) {
                        if (inner?.clobTokenIds) {
                            try {
                                const arr = typeof inner.clobTokenIds === 'string' ? JSON.parse(inner.clobTokenIds) : inner.clobTokenIds;
                                if (Array.isArray(arr) && arr.length >= 2) {
                                    clobs = arr;
                                    clobCount++;
                                    break;
                                }
                            } catch (_) {}
                        }
                    }
                }

                if (!newMarkets[asset][parsed.timeframe] || 
                    new Date(parsed.timestamp) > new Date(newMarkets[asset][parsed.timeframe].timestamp)) {
                    newMarkets[asset][parsed.timeframe] = {
                        slug,
                        title: event.title,
                        clobTokenIds: clobs,
                        timestamp: parsed.timestamp
                    };
                }
            }
        }
                
        // Pour chaque asset/timeframe manquant, essayer de récupérer par slug
        for (const asset of ASSETS) {
            for (const tf of TIMEFRAMES) {
                if (!newMarkets[asset][tf] || !newMarkets[asset][tf].clobTokenIds) {
                    const expectedSlug = generateExpectedSlug(asset, tf, now);
                    if (expectedSlug) {
                        try {
                            const response = await fetch(`https://gamma-api.polymarket.com/events/slug/${expectedSlug}`);
                            if (response.ok) {
                                const event = await response.json();
                                if (event && event.markets && Array.isArray(event.markets) && event.markets.length > 0) {
                                    for (const inner of event.markets) {
                                        if (inner?.clobTokenIds) {
                                            try {
                                                const arr = typeof inner.clobTokenIds === 'string' ? JSON.parse(inner.clobTokenIds) : inner.clobTokenIds;
                                                if (Array.isArray(arr) && arr.length >= 2) {
                                                    // Extraire le timestamp depuis le slug
                                                    const parsed = parseSlug(expectedSlug, asset);
                                                    if (parsed) {
                                                        newMarkets[asset][tf] = {
                                                            slug: expectedSlug,
                                                            title: event.title,
                                                            clobTokenIds: arr,
                                                            timestamp: parsed.timestamp
                                                        };
                                                        clobCount++;
                                                        activeCount++;
                                                        break;
                                                    }
                                                }
                                            } catch (_) {}
                                        }
                                    }
                                }
                            }
                        } catch (err) {
                            // Market doesn't exist yet, skip
                        }
                    }
                }
            }
        }
        
        // Mettre à jour MARKETS
        let hasChanges = false;
        ASSETS.forEach(asset => {
            TIMEFRAMES.forEach(tf => {
                const oldMarket = MARKETS[asset][tf];
                const newMarket = newMarkets[asset][tf];
                
                if ((!oldMarket && newMarket) || 
                    (oldMarket && newMarket && oldMarket.slug !== newMarket.slug)) {
                    MARKETS[asset][tf] = newMarket;
                    ACTIVE_BOUGIES[asset][tf] = newMarket?.timestamp;
                    hasChanges = true;
                    
                    // Log uniquement si on a les CLOB (prochain pari exploitable)
                    if (newMarket?.clobTokenIds && newMarket.clobTokenIds.length >= 2) {
                        const colorFn = ASSET_COLORS[asset] || colors.white;
                        const assetStr = colorFn(`[${asset}]`);
                        const marketTitle = newMarket?.title || newMarket?.slug || 'N/A';
                        console.log(`${assetStr} ${colors.green('✓')} ${colors.cyan(tf)}: ${colors.white(marketTitle)}`);
                        
                        if (DEBUG_MODE) {
                            console.log(`  ${colors.gray('API:')} https://gamma-api.polymarket.com/events/slug/${newMarket.slug}`);
                            console.log(`  ${colors.gray('PM:')}  https://polymarket.com/event/${newMarket.slug}`);
                        }
                    }
                }
            });
        });

        // No summary log

    } catch (error) {
        console.error('✗ Erreur refreshMarkets:', error.message);
    }
}

/**
 * Récupère le prix spot depuis Binance
 */
async function getSpotPrice(asset) {
    try {
        const symbol = `${asset}USDT`;
        const response = await fetch(`https://api.binance.com/api/v3/ticker/price?symbol=${symbol}`);
        const data = await response.json();
        return parseFloat(data.price);
    } catch (error) {
        console.error(`✗ Erreur prix spot ${asset}:`, error.message);
        return null;
    }
}

/**
 * Récupère les cotations CLOB Polymarket en masse
 * @param {Array} requests - Array of {token_id, side}
 * @returns {Object} - Map of "token_id-side" -> price
 */
async function getCLOBPricesBatch(requests) {
    try {
        const response = await fetch('https://clob.polymarket.com/prices', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requests)
        });
        const data = await response.json();
        
        // Construire un map des résultats
        // Format de réponse: { "token_id": { "BUY": "1800.50", "SELL": "1801.00" } }
        const priceMap = {};
        if (data && typeof data === 'object') {
            Object.entries(data).forEach(([tokenId, sides]) => {
                if (sides && typeof sides === 'object') {
                    Object.entries(sides).forEach(([side, priceStr]) => {
                        const key = `${tokenId}-${side}`;
                        const price = parseFloat(priceStr);
                        priceMap[key] = isNaN(price) ? null : price;
                    });
                }
            });
        }
        return priceMap;
    } catch (error) {
        // Retourner un map vide en cas d'erreur
        return {};
    }
}

/**
 * Récupère une seule cotation CLOB Polymarket (legacy, pour compatibilité)
 */
async function getCLOBPrice(clobTokenId, side) {
    const priceMap = await getCLOBPricesBatch([{ token_id: clobTokenId, side }]);
    const key = `${clobTokenId}-${side}`;
    return priceMap[key] || null;
}

/**
 * Collecte les données pour tous les assets
 */
async function collectData() {
    const now = new Date();
    let needRefresh = false;
    
    // Étape 1: Récupérer tous les prix spot et préparer les requêtes CLOB
    const batchRequests = [];
    const rows = {};
    
    for (const asset of ASSETS) {
        const spotPrice = await getSpotPrice(asset);
        if (!spotPrice) continue;

        rows[asset] = {
            timestamp: now.toISOString(),
            spot_price: spotPrice,
            hasAnyValidQuotes: false
        };

        // Préparer les requêtes CLOB pour cet asset
        for (const tf of TIMEFRAMES) {
            const market = MARKETS[asset][tf];
            
            if (!market || !market.clobTokenIds || market.clobTokenIds.length < 2) {
                rows[asset][`${tf}_buy`] = '';
                rows[asset][`${tf}_sell`] = '';
                rows[asset][`${tf}_spread_up`] = '';
                rows[asset][`${tf}_spread_down`] = '';
                continue;
            }

            // Vérifier que le marché correspond encore à la bougie active
            const parsed = parseSlug(market.slug, asset);
            if (!parsed) {
                rows[asset][`${tf}_buy`] = '';
                rows[asset][`${tf}_sell`] = '';
                rows[asset][`${tf}_spread_up`] = '';
                rows[asset][`${tf}_spread_down`] = '';
                continue;
            }
            
            const [isActive] = isActiveMarket(parsed, asset, tf, now);
            if (!isActive) {
                rows[asset][`${tf}_buy`] = '';
                rows[asset][`${tf}_sell`] = '';
                rows[asset][`${tf}_spread_up`] = '';
                rows[asset][`${tf}_spread_down`] = '';
                needRefresh = true;
                continue;
            }

            // Ajouter les requêtes pour ce marché
            const upTokenId = market.clobTokenIds[0];
            const downTokenId = market.clobTokenIds[1];
            batchRequests.push(
                { token_id: upTokenId, side: 'BUY' },
                { token_id: upTokenId, side: 'SELL' },
                { token_id: downTokenId, side: 'BUY' },
                { token_id: downTokenId, side: 'SELL' }
            );
        }
    }
    
    // Étape 2: Faire UNE seule requête batch pour tous les prix
    const priceMap = batchRequests.length > 0 ? await getCLOBPricesBatch(batchRequests) : {};
    
    // Étape 3: Parser les résultats et remplir les rows
    for (const asset of ASSETS) {
        const row = rows[asset];
        if (!row) continue;

        for (const tf of TIMEFRAMES) {
            const market = MARKETS[asset][tf];
            
            if (!market || !market.clobTokenIds || market.clobTokenIds.length < 2) {
                continue;
            }

            const parsed = parseSlug(market.slug, asset);
            if (!parsed) continue;
            
            const [isActive] = isActiveMarket(parsed, asset, tf, now);
            if (!isActive) continue;

            // Récupérer les prix depuis le map
            const upTokenId = market.clobTokenIds[0];
            const downTokenId = market.clobTokenIds[1];
            
            const upBuyPrice = priceMap[`${upTokenId}-BUY`] ?? null;
            const upSellPrice = priceMap[`${upTokenId}-SELL`] ?? null;
            const downBuyPrice = priceMap[`${downTokenId}-BUY`] ?? null;
            const downSellPrice = priceMap[`${downTokenId}-SELL`] ?? null;

            if (upBuyPrice !== null && downSellPrice !== null) {
                const spreadUp = (upSellPrice !== null && upBuyPrice !== null)
                    ? (upSellPrice - upBuyPrice) : null;
                const spreadDown = (downSellPrice !== null && downBuyPrice !== null) 
                    ? (downSellPrice - downBuyPrice) : null;
                
                row[`${tf}_buy`] = upBuyPrice.toFixed(2);
                row[`${tf}_sell`] = downSellPrice.toFixed(2);
                row[`${tf}_spread_up`] = spreadUp !== null ? spreadUp.toFixed(2) : '';
                row[`${tf}_spread_down`] = spreadDown !== null ? spreadDown.toFixed(2) : '';
                row.hasAnyValidQuotes = true;
            } else {
                row[`${tf}_buy`] = '';
                row[`${tf}_sell`] = '';
                row[`${tf}_spread_up`] = '';
                row[`${tf}_spread_down`] = '';
            }
        }

        // Ajouter au buffer si on a au moins un pari valide
        if (row.hasAnyValidQuotes) {
            delete row.hasAnyValidQuotes; // Nettoyer avant de push
            BUFFERS[asset].push(row);
        }
    }
    
    // Si un marché n'était plus actif, rafraîchir les marchés
    if (needRefresh) {
        await refreshMarkets();
    }
}

/**
 * Écrit les buffers dans les fichiers CSV
 */
async function flushToCSV() {
    console.log(colors.magenta('[FLUSH] Écriture dans les CSV...'));
    
    for (const asset of ASSETS) {
        const buffer = BUFFERS[asset];
        if (buffer.length === 0) continue;

        const csvPath = path.join(DATA_DIR, `${asset}.csv`);
        const lines = [];

        // Si le fichier n'existe pas, ajouter l'en-tête
        if (!fs.existsSync(csvPath)) {
            const header = 'timestamp,spot_price,m15_buy,m15_sell,m15_spread_up,m15_spread_down,h1_buy,h1_sell,h1_spread_up,h1_spread_down,daily_buy,daily_sell,daily_spread_up,daily_spread_down';
            lines.push(header);
        }

        // Ajouter les nouvelles lignes
        for (const row of buffer) {
            const cleanValue = (val) => (val && val !== 'NaN' && val !== null) ? val : '';
            // XRP utilise 4 décimales pour le prix spot
            const spotDecimals = asset === 'XRP' ? 4 : 2;
            const line = [
                row.timestamp,
                row.spot_price.toFixed(spotDecimals),
                cleanValue(row.m15_buy),
                cleanValue(row.m15_sell),
                cleanValue(row.m15_spread_up),
                cleanValue(row.m15_spread_down),
                cleanValue(row.h1_buy),
                cleanValue(row.h1_sell),
                cleanValue(row.h1_spread_up),
                cleanValue(row.h1_spread_down),
                cleanValue(row.daily_buy),
                cleanValue(row.daily_sell),
                cleanValue(row.daily_spread_up),
                cleanValue(row.daily_spread_down)
            ].join(',');
            lines.push(line);
        }

        // Écrire dans le fichier
        fs.appendFileSync(csvPath, lines.join('\n') + '\n');
        
        // Log coloré par asset
        const colorFn = ASSET_COLORS[asset] || colors.white;
        const assetStr = colorFn(`[${asset}]`);
        console.log(`${assetStr} ${colors.green('✓')} Écrit ${colors.yellow(buffer.length)} lignes dans ${colors.white(asset + '.csv')}`);

        // Vider le buffer
        BUFFERS[asset] = [];
    }
    
    console.log(colors.green('[FLUSH] ✓ Terminé\n'));
}

/**
 * Fonction principale
 */
async function main() {
    console.log(colors.red('Polymarket Price Logger démarré\n'));
    
    // Assets avec couleurs
    const coloredAssets = ASSETS.map(asset => ASSET_COLORS[asset](asset)).join(', ');
    console.log(`${colors.bold('Assets:')} ${coloredAssets}`);
    console.log(`${colors.bold('Timeframes:')} ${colors.cyan(TIMEFRAMES.join(', '))}`);
    console.log(`${colors.bold('Dossier de sortie:')} ${colors.yellow(DATA_DIR)}`);
    if (DEBUG_MODE) {
        console.log(`${colors.bold('Mode:')} ${colors.yellow('DEBUG')} (affichage des URLs)\n`);
    }

    // Référencement initial des marchés
    await refreshMarkets();

    // Essayer d'obtenir tous les timeframes au démarrage (quelques tentatives rapides)
    let tries = 0;
    const needAll = () => ASSETS.every(a => TIMEFRAMES.every(tf => MARKETS[a][tf]?.clobTokenIds && MARKETS[a][tf].clobTokenIds.length >= 2));
    while (!needAll() && tries < 5) {
        await new Promise(r => setTimeout(r, 2000));
        await refreshMarkets();
        tries += 1;
    }

    // Tick toutes les secondes
    setInterval(async () => {
        await collectData();
    }, 1000);

    // Flush toutes les 60 secondes
    setInterval(async () => {
        await flushToCSV();
    }, 60000);

    console.log(colors.green('✓ Logging démarré') + colors.gray(' (tick: 1s, flush: 60s)\n'));
}

// Gestion des erreurs non catchées
process.on('unhandledRejection', (error) => {
    console.error('✗ Unhandled Rejection:', error);
});

// Démarrage
main().catch(console.error);

