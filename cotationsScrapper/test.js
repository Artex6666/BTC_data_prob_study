const fetch = require('node-fetch');

// Configuration
const ASSET = 'BTC';
const TIMEFRAME = 'm15';

// Structure de stockage du marché
let currentMarket = null;

/**
 * Parse le slug et extrait les informations de la bougie
 */
function parseSlug(slug, asset) {
    const shortName = asset.toLowerCase();
    
    // Format: btc-updown-15m-1762104600 (timestamp Unix)
    const unixPattern = new RegExp(`(${shortName})-updown-(15m|1h|1d)-(\\d+)`);
    const unixMatch = slug.match(unixPattern);
    
    if (unixMatch) {
        const unixTimestamp = parseInt(unixMatch[3]);
        const timestamp = new Date(unixTimestamp * 1000);
        return { timestamp };
    }
    
    return null;
}

/**
 * Vérifie si le marché correspond à la bougie active
 */
function isActiveMarket(parsed, asset, timeframe, now = new Date()) {
    // Convertir now et timestamp en composants ET
    const formatParts = (date) => {
        const str = date.toLocaleString('en-US', { timeZone: 'America/New_York' });
        const match = str.match(/(\d+)\/(\d+)\/(\d+), (\d+):(\d+):(\d+)/);
        if (!match) return null;
        const month = parseInt(match[1]);
        const day = parseInt(match[2]);
        const year = parseInt(match[3]);
        const hour = parseInt(match[4]);
        const minute = parseInt(match[5]);
        return { year, month, day, hour, minute };
    };

    const get = (parts, key) => {
        const idx = { year: 3, month: 1, day: 2, hour: 4, minute: 5 };
        const match = parts.match(/(\d+)\/(\d+)\/(\d+), (\d+):(\d+):(\d+)/);
        if (!match) return null;
        return parseInt(match[idx[key]]);
    };

    const nowParts = now.toLocaleString('en-US', { timeZone: 'America/New_York' });
    const mParts = parsed.timestamp.toLocaleString('en-US', { timeZone: 'America/New_York' });

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

    // Pour m15, vérifier que now est dans les 15 minutes de la bougie
    const matches = nowY === mY && nowMo === mMo && nowD === mD && nowH === mH && 
                  nowMi >= mMi && nowMi < mMi + 15;

    return [matches, parsed.timestamp];
}

/**
 * Récupère le marché BTC m15 actuel
 */
async function refreshMarket() {
    try {
        const response = await fetch(
            `https://gamma-api.polymarket.com/events?closed=false&limit=100&order=id&ascending=false`
        );
        const data = await response.json();

        const markets = data.events || data;

        if (!markets || !Array.isArray(markets)) {
            console.log('✗ Réponse invalide de l\'API Gamma');
            return;
        }

        const now = new Date();

        for (const event of markets) {
            const slug = event?.slug;
            if (!slug) continue;

            // Filtrer BTC uniquement
            if (!slug.startsWith('btc-')) continue;

            // Déterminer timeframe
            let tfHint = null;
            try {
                const rec = Array.isArray(event.series) && event.series.length > 0 
                    ? (event.series[0]?.recurrence || null) : null;
                if (rec) {
                    const r = String(rec).toLowerCase();
                    if (r.includes('15')) tfHint = 'm15';
                    else if (r.includes('1h') || r.includes('hour')) tfHint = 'h1';
                    else if (r.includes('day') || r.includes('daily') || r.includes('1d')) tfHint = 'daily';
                }
            } catch (_) {}

            const parsed = parseSlug(slug, ASSET, tfHint);
            if (!parsed) continue;

            // Vérifier que c'est la bougie active m15
            if (tfHint !== 'm15') continue;
            const [isActive] = isActiveMarket(parsed, ASSET, 'm15', now);

            if (isActive) {
                // Extraire clobTokenIds
                let clobs = null;
                if (Array.isArray(event.markets) && event.markets.length > 0) {
                    for (const inner of event.markets) {
                        if (inner?.clobTokenIds) {
                            try {
                                const arr = typeof inner.clobTokenIds === 'string' 
                                    ? JSON.parse(inner.clobTokenIds) : inner.clobTokenIds;
                                if (Array.isArray(arr) && arr.length >= 2) {
                                    clobs = arr;
                                    break;
                                }
                            } catch (_) {}
                        }
                    }
                }

                if (clobs && clobs.length >= 2) {
                    currentMarket = {
                        slug,
                        title: event.title || slug,
                        clobTokenIds: clobs
                    };
                    console.log(`\n✓ Marché trouvé: ${currentMarket.title}`);
                    return;
                }
            }
        }
    } catch (error) {
        console.error('✗ Erreur refreshMarket:', error.message);
    }
}

/**
 * Récupère le prix CLOB Polymarket
 */
async function getCLOBPrice(clobTokenId, side) {
    try {
        const response = await fetch(
            `https://clob.polymarket.com/price?token_id=${clobTokenId}&side=${side}`
        );
        const data = await response.json();
        const price = parseFloat(data.price);
        return isNaN(price) ? null : price;
    } catch (error) {
        return null;
    }
}

/**
 * Log les prix toutes les secondes
 */
async function logPrices() {
    while (true) {
        if (!currentMarket || !currentMarket.clobTokenIds) {
            console.log('⏳ Attente du marché BTC m15...');
            await refreshMarket();
            await new Promise(resolve => setTimeout(resolve, 1000));
            continue;
        }

        const upTokenId = currentMarket.clobTokenIds[0];
        const downTokenId = currentMarket.clobTokenIds[1];

        const upBuy = await getCLOBPrice(upTokenId, 'BUY');
        const downBuy = await getCLOBPrice(downTokenId, 'BUY');

        const now = new Date().toISOString();
        console.log(`${now} | UP Buy: ${upBuy !== null ? upBuy.toFixed(2) : 'N/A'} | DOWN Buy: ${downBuy !== null ? downBuy.toFixed(2) : 'N/A'}`);

        // Vérifier si le marché est toujours actif
        const parsed = parseSlug(currentMarket.slug, ASSET);
        if (parsed) {
            const [isActive] = isActiveMarket(parsed, ASSET, 'm15');
            if (!isActive) {
                console.log('⏳ Marché inactif, recherche du nouveau marché...');
                currentMarket = null;
                await refreshMarket();
            }
        }

        await new Promise(resolve => setTimeout(resolve, 1000));
    }
}

// Lancer le script
async function main() {
    console.log('🚀 Démarrage du test BTC m15...\n');
    await refreshMarket();
    logPrices();
}

main().catch(console.error);

