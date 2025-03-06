import {env as nodeEnv} from 'node:process';
import {z} from 'zod';

const zEnv = z.object({
    TRANS_FRONT_PATH: z.string().default('../../front/dist/'),
    TRANS_VIEWS_PATH: z.string().default('../src/static/views/'),
    TRANS_TAIL_PATH: z.string().default('../src/static/tail/'),
    TRANS_ICO_PATH: z.string().default('../src/static/favicon/'),
})
const env = zEnv.parse(nodeEnv);

export {env}