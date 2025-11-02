const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');
const colors = require('colors');

// Configuration
const ASSETS = ['BTC', 'ETH', 'SOL', 'XRP'];
const TIMEFRAMES = ['m15', 'h1', 'daily'];
const DATA_DIR = path.join(__dirname, '..', 'data');

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
function parseSlug(slug, asset) {
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
            // On convertit d'abord en UTC en utilisant toLocaleString avec timeZone
            let dateStr;
            if (tf === 'm15') {
                dateStr = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}T${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:00`;
            } else if (tf === 'h1') {
                dateStr = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}T${String(hour).padStart(2, '0')}:00:00`;
            } else {
                dateStr = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}T23:59:59`;
            }
            
            // Créer une date UTILISANT l'API pour simuler ET->UTC
            // On crée une date naïve et on la convertit comme si c'était ET
            const etDateStr = `${dateStr}-05:00`; // Assume EST (gérer EDT séparément si nécessaire)
            
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
    const etFormatter = new Intl.DateTimeFormat('en-US', {
        timeZone: 'America/New_York',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
    });
    
    const etParts = etFormatter.formatToParts(now);
    const year = parseInt(etParts.find(p => p.type === 'year').value);
    const month = parseInt(etParts.find(p => p.type === 'month').value) - 1;
    const day = parseInt(etParts.find(p => p.type === 'day').value);
    const hour = parseInt(etParts.find(p => p.type === 'hour').value);
    const minute = parseInt(etParts.find(p => p.type === 'minute').value);
    
    // Créer un timestamp en ET
    const etDate = new Date(Date.UTC(year, month, day, hour, minute));
    
    let targetET;
    if (timeframe === 'm15') {
        // Prochaine bougie 15 minutes
        const next15 = Math.ceil(minute / 15) * 15;
        targetET = new Date(Date.UTC(year, month, day, hour, next15));
        if (next15 === 60) {
            targetET = new Date(Date.UTC(year, month, day, hour + 1, 0));
        }
    } else if (timeframe === 'h1') {
        // Prochaine bougie heure
        targetET = new Date(Date.UTC(year, month, day, hour + 1, 0));
    } else if (timeframe === 'daily') {
        // Fin du jour actuel
        targetET = new Date(Date.UTC(year, month, day, 23, 59));
    }

    return targetET;
}

/**
 * Vérifie si un marché correspond à la bougie active
 */
function isActiveMarket(parsed, asset, timeframe, now = new Date()) {
    if (!parsed) return false;
    
    const activeBougie = getActiveBougie(asset, timeframe, now);
    
    // Convertir les deux timestamps en composants ET
    const etFormatter = new Intl.DateTimeFormat('en-US', {
        timeZone: 'America/New_York',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
    });
    
    const activeParts = etFormatter.formatToParts(activeBougie);
    const marketParts = etFormatter.formatToParts(parsed.timestamp);
    
    const activeYear = parseInt(activeParts.find(p => p.type === 'year').value);
    const activeMonth = parseInt(activeParts.find(p => p.type === 'month').value);
    const activeDay = parseInt(activeParts.find(p => p.type === 'day').value);
    const activeHour = parseInt(activeParts.find(p => p.type === 'hour').value);
    const activeMinute = parseInt(activeParts.find(p => p.type === 'minute').value);
    
    const marketYear = parseInt(marketParts.find(p => p.type === 'year').value);
    const marketMonth = parseInt(marketParts.find(p => p.type === 'month').value);
    const marketDay = parseInt(marketParts.find(p => p.type === 'day').value);
    const marketHour = parseInt(marketParts.find(p => p.type === 'hour').value);
    const marketMinute = parseInt(marketParts.find(p => p.type === 'minute').value);
    
    // Comparaison au niveau de la résolution appropriée
    if (timeframe === 'm15') {
        return activeYear === marketYear && 
               activeMonth === marketMonth && 
               activeDay === marketDay && 
               activeHour === marketHour && 
               activeMinute === marketMinute;
    } else if (timeframe === 'h1') {
        return activeYear === marketYear && 
               activeMonth === marketMonth && 
               activeDay === marketDay && 
               activeHour === marketHour;
    } else if (timeframe === 'daily') {
        return activeYear === marketYear && 
               activeMonth === marketMonth && 
               activeDay === marketDay;
    }
    
    return false;
}

/**
 * Récupère les marchés Polymarket et met à jour les clobTokenIds
 */
async function refreshMarkets() {
    console.log(colors.cyan('\n[REFRESH MARKETS] Démarrage...'));
    
    try {
        const response = await fetch('https://gamma-api.polymarket.com/events?closed=false&limit=500&order=id&ascending=false');
        const data = await response.json();
        
        if (!data || !Array.isArray(data)) {
            console.log('✗ Réponse invalide de l\'API Gamma');
            return;
        }

        const now = new Date();
        const newMarkets = {};

        ASSETS.forEach(asset => {
            newMarkets[asset] = { m15: null, h1: null, daily: null };
        });

        // Filtrer et parser les marchés
        for (const market of data) {
            if (!market.slug || !market.clobTokenIds || !market.outcomes) continue;
            
            const parsed = parseSlug(market.slug, 'BTC') || 
                          parseSlug(market.slug, 'ETH') || 
                          parseSlug(market.slug, 'SOL') || 
                          parseSlug(market.slug, 'XRP');
            
            if (!parsed) continue;
            
            let asset;
            if (market.slug.includes('btc-updown')) asset = 'BTC';
            else if (market.slug.includes('eth-updown')) asset = 'ETH';
            else if (market.slug.includes('sol-updown')) asset = 'SOL';
            else if (market.slug.includes('xrp-updown')) asset = 'XRP';
            else continue;

            // Vérifier que c'est la bougie active
            if (isActiveMarket(parsed, asset, parsed.timeframe, now)) {
                if (!newMarkets[asset][parsed.timeframe] || 
                    new Date(parsed.timestamp) > new Date(newMarkets[asset][parsed.timeframe].timestamp)) {
                    newMarkets[asset][parsed.timeframe] = {
                        slug: market.slug,
                        title: market.title,
                        clobTokenIds: market.clobTokenIds,
                        timestamp: parsed.timestamp
                    };
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
                    
                    // Log coloré par asset
                    const colorFn = ASSET_COLORS[asset] || colors.white;
                    const assetStr = colorFn(`[${asset}]`);
                    const marketTitle = newMarket?.title || 'N/A';
                    console.log(`${assetStr} ${colors.green('✓')} ${colors.cyan(tf)}: ${colors.white(marketTitle)}`);
                }
            });
        });

        if (hasChanges) {
            console.log(colors.green('[REFRESH MARKETS] ✓ Mis à jour avec succès\n'));
        } else {
            console.log(colors.gray('[REFRESH MARKETS] ✓ Aucun changement\n'));
        }

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
 * Récupère les cotations CLOB Polymarket
 */
async function getCLOBPrice(clobTokenId, side) {
    try {
        const response = await fetch(`https://clob.polymarket.com/price?token_id=${clobTokenId}&side=${side}`);
        const data = await response.json();
        return parseFloat(data.price);
    } catch (error) {
        // Pas d'erreur visible, juste retour null
        return null;
    }
}

/**
 * Collecte les données pour tous les assets
 */
async function collectData() {
    const now = new Date();
    
    for (const asset of ASSETS) {
        // Récupérer le prix spot
        const spotPrice = await getSpotPrice(asset);
        if (!spotPrice) continue;

        // Préparer la ligne de données
        const row = {
            timestamp: now.toISOString(),
            spot_price: spotPrice
        };

        let hasAnyValidQuotes = false;

        // Récupérer les cotations pour chaque timeframe
        for (const tf of TIMEFRAMES) {
            const market = MARKETS[asset][tf];
            
            if (!market || !market.clobTokenIds || market.clobTokenIds.length < 2) {
                // Pas de marché disponible pour ce timeframe
                row[`${tf}_buy`] = '';
                row[`${tf}_sell`] = '';
                row[`${tf}_spread`] = '';
                continue;
            }

            // Récupérer les prix (on utilise le token Up)
            const upTokenId = market.clobTokenIds[0]; // Généralement le Up
            
            const buyPrice = await getCLOBPrice(upTokenId, 'BUY');
            const sellPrice = await getCLOBPrice(upTokenId, 'SELL');

            if (buyPrice !== null && sellPrice !== null) {
                const spread = sellPrice - buyPrice;
                row[`${tf}_buy`] = buyPrice.toFixed(6);
                row[`${tf}_sell`] = sellPrice.toFixed(6);
                row[`${tf}_spread`] = spread.toFixed(6);
                hasAnyValidQuotes = true;
            } else {
                row[`${tf}_buy`] = '';
                row[`${tf}_sell`] = '';
                row[`${tf}_spread`] = '';
            }
        }

        // N'ajouter au buffer que si on a au moins un pari valide
        if (hasAnyValidQuotes) {
            BUFFERS[asset].push(row);
        }
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
            const header = 'timestamp,spot_price,m15_buy,m15_sell,m15_spread,h1_buy,h1_sell,h1_spread,daily_buy,daily_sell,daily_spread';
            lines.push(header);
        }

        // Ajouter les nouvelles lignes
        for (const row of buffer) {
            const line = [
                row.timestamp,
                row.spot_price.toFixed(2),
                row.m15_buy || '',
                row.m15_sell || '',
                row.m15_spread || '',
                row.h1_buy || '',
                row.h1_sell || '',
                row.h1_spread || '',
                row.daily_buy || '',
                row.daily_sell || '',
                row.daily_spread || ''
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
    console.log(colors.rainbow('🚀 Polymarket Price Logger démarré\n'));
    
    // Assets avec couleurs
    const coloredAssets = ASSETS.map(asset => ASSET_COLORS[asset](asset)).join(', ');
    console.log(`${colors.bold('Assets:')} ${coloredAssets}`);
    console.log(`${colors.bold('Timeframes:')} ${colors.cyan(TIMEFRAMES.join(', '))}`);
    console.log(`${colors.bold('Dossier de sortie:')} ${colors.yellow(DATA_DIR)}\n`);

    // Référencement initial des marchés
    await refreshMarkets();

    // Tick toutes les secondes
    setInterval(async () => {
        await collectData();
    }, 1000);

    // Flush toutes les 60 secondes
    setInterval(async () => {
        await flushToCSV();
    }, 60000);

    // Refresh markets toutes les 10 minutes
    setInterval(async () => {
        await refreshMarkets();
    }, 600000);

    console.log(colors.green('✓ Logging démarré') + colors.gray(' (tick: 1s, flush: 60s, refresh: 10min)\n'));
}

// Gestion des erreurs non catchées
process.on('unhandledRejection', (error) => {
    console.error('✗ Unhandled Rejection:', error);
});

// Démarrage
main().catch(console.error);

