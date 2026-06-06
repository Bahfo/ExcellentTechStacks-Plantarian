const path = require("path");
require("dotenv").config({path: path.join(__dirname, "../.env")});

console.log("MONGO_URI =", process.env.MONGO_URI);
console.log("DB_NAME =", process.env.DB_NAME);

require ("dotenv").config();
const { MongoClient } = require("mongodb");

const client = new MongoClient(process.env.MONGO_URI);

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

async function createSplits() {
    await client.connect();
    const db = client.db(process.env.DB_NAME);
    const col = db.collection("images_metadata");
    const classes = await col.distinct("datasetClass");

    for (const datasetClass of classes) {
        const docs = await col.find({datasetClass}).toArray();
        shuffle(docs);
        const total = docs.length;
        const trainEnd = Math.floor(total * 0.70);
        const validationEnd = Math.floor(total * 0.85);

        for (let i = 0; i < total; i++) {
            let split;

            if (i < trainEnd) split = "train";
            else if (i < validationEnd) split = "validation";
            else split = "test";

            await col.updateOne({_id: docs[i]._id},
                {$set: {datasetSplit: split}});
        }

        console.log(`Processed ${datasetClass}`);
    }

    await client.close();
    console.log("Dataset split completed.");
}

createSplits().catch(console.error);