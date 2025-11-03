# Polymarket Price Logger

Script Node.js pour logger en continu les prix spot et les cotations Polymarket pour plusieurs actifs crypto.

## 📋 Fonctionnalités

- **Multi-assets** : BTC, ETH, SOL, XRP
- **Multi-timeframes** : m15, h1, daily
- **Collecte en continu** : Prix spot et cotations Polymarket toutes les secondes
- **Écriture périodique** : Sauvegarde CSV toutes les 60 secondes
- **Refresh automatique** : Mise à jour des marchés toutes les 10 minutes
- **Validation des bougies actives** : Vérification que les paris correspondent aux périodes actives

## 🚀 Installation

```bash
cd cotationsScrapper
npm install
```

## ▶️ Utilisation

```bash
npm start
```

Le script va :
1. Créer automatiquement le dossier `data/` à la racine du projet
2. Générer un fichier CSV par asset : `BTC.csv`, `ETH.csv`, `SOL.csv`, `XRP.csv`
3. Logger toutes les secondes et écrire toutes les minutes

## 📊 Format des données CSV

Chaque fichier CSV contient :

```csv
timestamp,spot_price,m15_buy,m15_sell,m15_spread,h1_buy,h1_sell,h1_spread,daily_buy,daily_sell,daily_spread
2025-11-02T22:00:00.123Z,67123.22,0.514,0.519,0.005,0.505,0.509,0.004,0.487,0.491,0.004
```

- **timestamp** : ISO 8601
- **spot_price** : Prix spot depuis Binance
- **m15_buy/sell/spread** : Cotations pour les paris 15 minutes
- **h1_buy/sell/spread** : Cotations pour les paris 1 heure
- **daily_buy/sell/spread** : Cotations pour les paris journaliers

## ⚙️ Configuration

Modifiable dans `polymarket_logger.js` :

```javascript
const ASSETS = ['BTC', 'ETH', 'SOL', 'XRP'];     // Actifs à suivre
const TIMEFRAMES = ['m15', 'h1', 'daily'];       // Timeframes
const DATA_DIR = path.join(__dirname, '..', 'data'); // Dossier de sortie
```

## 🧠 Fonctionnement

### 1. Refresh des marchés (10 minutes)
- Appel à `GET https://gamma-api.polymarket.com/events?closed=false&limit=100`
- Filtrage des événements updown pour chaque asset/timeframe
- Validation que le marché correspond à la bougie active (timezone ET)
- Mise à jour des `clobTokenIds` depuis `event.markets[].clobTokenIds`
- Fallback : récupération par slug pour les marchés manquants via `/events/slug/{slug}`

### 2. Collecte des données (1 seconde)
- Prix spot depuis Binance (`GET https://api.binance.com/api/v3/ticker/price`)
- Cotations Polymarket CLOB (`GET https://clob.polymarket.com/price`)
- Cotations pour les tokens Up et Down
- **Spread** = Up.sell - Up.buy (différence côté token Up) et Down.sell - Down.buy (différence côté token Down)
- **Buy** = Up SELL (prix pour vendre Up)
- **Sell** = Down SELL (prix pour vendre Down)

### 3. Écriture CSV (60 secondes)
- Écriture de toutes les lignes collectées
- Vidage du buffer

## 🔍 Validation des bougies actives

Le script vérifie **doublement** que chaque pari utilisé correspond bien au prix spot enregistré :

1. **Au refresh** (toutes les 10 minutes) : Filtrage initial des marchés pour ne garder que les bougies actives
2. **À la collecte** (chaque seconde) : Re-vérification que le marché est toujours actif avant de stocker les données

### Parsing des slugs
- **m15** : Format Unix timestamp `btc-updown-15m-1762120800`
- **h1** : Format texte `bitcoin-up-or-down-november-2-3pm-et`
- **daily** : Format texte `xrp-up-or-down-november-2-11pm-et`

### Calcul de la bougie active
- Timezone **ET (America/New_York)** avec gestion DST automatique
- **m15** : Prochaine bougie (00, 15, 30, 45 minutes)
- **h1** : Prochaine heure
- **daily** : 23:00 ET du jour actuel ou suivant

Exemple de title : `"Solana Up or Down - November 2, 1:30PM-1:45PM ET"`

## ⚠️ Gestion des erreurs

- Erreurs réseau : Loggées mais n'arrêtent pas le script
- Marchés indisponibles : Champs vides dans le CSV
- **Stockage conditionnel** : Les lignes ne sont sauvegardées que si au moins un pari actif (m15, h1 ou daily) est disponible
- API limits : Respecte les limites de taux

## 📝 Notes

- Les bougies utilisent la timezone **ET (America/New_York)**
- Pour m15 : bougie de 00, 15, 30, 45 minutes
- Pour h1 : bougie de chaque heure
- Pour daily : bougie de minuit à minuit ET

