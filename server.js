const util = require('util');

if (!util.isNullOrUndefined)
    util.isNullOrUndefined = v => v === null || v === undefined;

if (!util.isArray)
    util.isArray = Array.isArray;

const express = require('express');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const mongoose = require('mongoose');
const session = require('cookie-session');
const User = require('./models/User');

const app = express();
const PORT = 8000;
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/plantarian_db';

// Trust Nginx Proxy
app.set('trust proxy', 1);

// Database Connection
mongoose.connect(MONGO_URI)
    .then(() => console.log('Connected to MongoDB'))
    .catch(err => console.error('MongoDB connection error:', err));

app.use(express.json({ limit: '50mb' }));

// Session Middleware
app.use(session({
    name: 'session',
    keys: ['plantarian-secret-key-123'],
    maxAge: 24 * 60 * 60 * 1000 // 24 hours
}));

// Auth Middleware
const requireAuth = (req, res, next) => {
    console.log(`[AUTH] Check for ${req.url} - User ID: ${req.session.userId}`);
    if (!req.session.userId) {
        if (req.xhr || req.headers.accept.indexOf('json') > -1) {
            return res.status(401).json({ error: 'Unauthorized' });
        }
        return res.redirect('/login');
    }
    next();
};

// Static files - serve files but protect the root
app.use(express.static(__dirname, { index: false }));

app.get('/login', (req, res) => {
    res.sendFile(path.join(__dirname, 'login.html'));
});

app.get('/', (req, res) => {
    if (req.session.userId) {
        return res.redirect('/dashboard');
    }
    res.sendFile(path.join(__dirname, 'welcome.html'));
});

app.get('/dashboard', requireAuth, (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Auth API Routes
app.post('/api/auth/register', async (req, res) => {
    try {
        const { username, email, password } = req.body;
        console.log(`[AUTH] Registering user: ${username}`);
        const user = new User({ username, email, password });
        await user.save();
        req.session.userId = user._id;
        console.log(`[AUTH] Registration success: ${user._id}`);
        res.json({ success: true, user: { username: user.username } });
    } catch (err) {
        console.error('[AUTH] Registration error:', err.message);
        res.status(400).json({ error: err.message });
    }
});

app.post('/api/auth/login', async (req, res) => {
    try {
        const { username, password } = req.body;
        console.log(`[AUTH] Login attempt: ${username}`);
        const user = await User.findOne({ username });
        if (!user || !(await user.comparePassword(password))) {
            console.log('[AUTH] Login failed: Invalid credentials');
            return res.status(401).json({ error: 'Invalid credentials' });
        }
        req.session.userId = user._id;
        console.log(`[AUTH] Login success: ${user._id}`);
        res.json({ success: true, user: { username: user.username } });
    } catch (err) {
        console.error('[AUTH] Login error:', err.message);
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/auth/logout', (req, res) => {
    req.session = null;
    res.redirect('/login');
});

app.get('/api/auth/me', async (req, res) => {
    if (!req.session.userId) return res.status(401).json({ error: 'Not logged in' });
    const user = await User.findById(req.session.userId);
    res.json({ username: user.username, email: user.email });
});

const CLASS_LABELS = {
    0: "Apple: Apple scab",
    1: "Apple: Black rot",
    2: "Apple: Cedar apple rust",
    3: "Apple: Healthy",
    4: "Blueberry: Healthy",
    5: "Cherry: Powdery mildew",
    6: "Cherry: Healthy",
    7: "Corn: Cercospora leaf spot / Gray leaf spot",
    8: "Corn: Common rust",
    9: "Corn: Northern Leaf Blight",
    10: "Corn: Healthy",
    11: "Grape: Black rot",
    12: "Grape: Esca (Black Measles)",
    13: "Grape: Leaf blight (Isariopsis Leaf Spot)",
    14: "Grape: Healthy",
    15: "Orange: Haunglongbing (Citrus greening)",
    16: "Peach: Bacterial spot",
    17: "Peach: Healthy",
    18: "Pepper, bell: Bacterial spot",
    19: "Pepper, bell: Healthy",
    20: "Potato: Early blight",
    21: "Potato: Late blight",
    22: "Potato: Healthy",
    23: "Raspberry: Healthy",
    24: "Soybean: Healthy",
    25: "Squash: Powdery mildew",
    26: "Strawberry: Leaf scorch",
    27: "Strawberry: Healthy",
    28: "Tomato: Bacterial spot",
    29: "Tomato: Early blight",
    30: "Tomato: Late blight",
    31: "Tomato: Leaf Mold",
    32: "Tomato: Septoria leaf spot",
    33: "Tomato: Spider mites (Two-spotted spider mite)",
    34: "Tomato: Target Spot",
    35: "Tomato: Tomato Yellow Leaf Curl Virus",
    36: "Tomato: Tomato mosaic virus",
    37: "Tomato: Healthy"
};

const CLASS_TO_SPECIES = {
    0: "Apple", 1: "Apple", 2: "Apple", 3: "Apple", 4: "Blueberry", 5: "Cherry", 6: "Cherry", 7: "Corn", 8: "Corn", 9: "Corn", 10: "Corn",
    11: "Grape", 12: "Grape", 13: "Grape", 14: "Grape", 15: "Orange", 16: "Peach", 17: "Peach", 18: "Pepper, bell", 19: "Pepper, bell",
    20: "Potato", 21: "Potato", 22: "Potato", 23: "Raspberry", 24: "Soybean", 25: "Squash", 26: "Strawberry", 27: "Strawberry",
    28: "Tomato", 29: "Tomato", 30: "Tomato", 31: "Tomato", 32: "Tomato", 33: "Tomato", 34: "Tomato", 35: "Tomato", 36: "Tomato", 37: "Tomato"
};

let model;

async function preprocess(base64) {
    const imageBuffer = Buffer.from(base64.replace(/^data:image\/\w+;base64,/, ''), 'base64');
    
    // Match Python 'powerful_preprocess': Resize -> CLAHE (on L channel) -> Gaussian Blur
    const rgb = await sharp(imageBuffer)
        .resize(224, 224)
        .clahe({ width: 8, height: 8, maxSlope: 3 })
        .blur(0.8)
        .removeAlpha()
        .raw()
        .toBuffer();
    
    return tf.tensor4d(Float32Array.from(rgb, x => (x / 127.5) - 1), [1, 224, 224, 3]);
}

async function inference(input) {
    try {
        const output = model.execute({ input_layer_1: input }, 'Identity');
        return Array.isArray(output) ? output[0] : output;
    } catch {
        const output = model.execute(input);
        return Array.isArray(output) ? output[0] : output;
    }
}

app.post('/predict', requireAuth, async (req, res) => {
    let input = null;
    let output = null;
    try {
        if (!req.body?.image) return res.status(400).json({ error: 'No image supplied' });
        input = await preprocess(req.body.image);
        output = await inference(input);
        const probs = await output.data();
        let idx = 0;
        for (let i = 1; i < probs.length; i++) if (probs[i] > probs[idx]) idx = i;
        res.json({
            species: CLASS_TO_SPECIES[idx] ?? 'Unknown',
            condition: CLASS_LABELS[idx] ?? 'Unknown',
            confidence: (probs[idx] * 100).toFixed(2)
        });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    } finally {
        output?.dispose();
        input?.dispose();
    }
});

console.log('Starting server...');
(async () => {
    try {
        console.log('Loading model...');
        model = await tf.loadGraphModel(`file://${path.join(__dirname, 'web_model', 'model.json')}`);
        console.log('Model loaded successfully');
        
        console.log(`Starting app on port ${PORT}...`);
        app.listen(PORT, '0.0.0.0', () => {
            console.log(`Server running on port ${PORT} (0.0.0.0)`);
        });
    } catch (err) {
        console.error('CRITICAL STARTUP ERROR:', err);
    }
})();
