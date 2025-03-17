import {env as nodeEnv} from 'node:process';
// @ts-ignore
import { z } from "zod";

const zEnv = z.object({
    TRANS_FRONT_PATH: z.string().default('../public/'),
    TRANS_VIEWS_PATH: z.string().default('../public/views/'),
    TRANS_ICO_PATH: z.string().default('../public/'),
    TRANS_TAIL_PATH: z.string().default('../public/'),
})
const env = zEnv.parse(nodeEnv);

export {env}