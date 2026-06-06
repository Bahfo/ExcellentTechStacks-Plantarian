const { createDataset } = require("./scripts/dataloader");

(async () => {
    const ds = await createDataset("train");

    const batch = await ds.take(1).toArray();

    console.log("Batch loaded successfully");
    console.log(batch[0]);
})();