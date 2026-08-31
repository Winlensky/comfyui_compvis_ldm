import { app } from "../../scripts/app.js";

const STYLE = {
    LDMCheckpointLoader: { color: "#3a2b52", bgcolor: "#2a1d3d" },
    LDMBERTTextEncode:   { color: "#2b3d52", bgcolor: "#1d2a3d" },
    LDMEmptyLatent:      { color: "#2b5239", bgcolor: "#1d3d28" },
    LDMSampler:          { color: "#1f4a44", bgcolor: "#15332f" },
    LDMVAEDecode:        { color: "#4a3a1f", bgcolor: "#332815" },
	LDMVAEEncode:        { color: "#3d4a1f", bgcolor: "#2b3315" },
	LDMSaveCheckpoint:   { color: "#4a1f2b", bgcolor: "#33151d" },
};

app.registerExtension({
    name: "LDM.Legacy.NodeStyle",
    nodeCreated(node) {
        const s = STYLE[node.comfyClass];
        if (s) {
            node.color = s.color;
            node.bgcolor = s.bgcolor;
        }
    },
});