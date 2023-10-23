#pragma once
#ifndef _BOX_OP_EULER_
#define _BOX_OP_EULER_
#include "Proto.H"
#include "shim.hpp"
#include <chrono>
// #include "hip/hip_runtime.h"
// #include "cuda.h"
// #include "cuda_runtime.h"
#define NUMCOMPS DIM+2

// __device__ double P1[145808];
// __device__ double P2[278768];

// void init_full_euler_gpu() {
// }

// __global__ void ker_code0(const double *X, double gamma1) {
//     if (((((256*blockIdx.x) + threadIdx.x) < 18496))) {
//         double a239, a240, s24, s25, s26;
//         int a235, a236, a237, a238;
//         a235 = (threadIdx.x + (256*blockIdx.x));
//         s24 = X[a235];
//         P1[a235] = s24;
//         a236 = (a235 + 18496);
//         s25 = (X[a236] / s24);
//         P1[a236] = s25;
//         a237 = (a235 + 36992);
//         s26 = (X[a237] / s24);
//         P1[a237] = s26;
//         a238 = (a235 + 55488);
//         a239 = (gamma1 - 1.0);
//         a240 = (0.5*a239);
//         P1[a238] = ((a239*X[a238]) - ((a240*((s25*s25)*s24)) + (a240*((s26*s26)*s24))));
//     }
// }

// __global__ void ker_code1(const double *X, double gamma1) {
//     if (((((256*blockIdx.x) + threadIdx.x) < 17956))) {
//         double a468, a469, s93, s94, s95, s96, s97, s98, 
//                 t110;
//         int a467, b57;
//         a467 = ((256*blockIdx.x) + threadIdx.x);
//         b57 = ((136*(a467 / 134)) + (((122*blockIdx.x) + threadIdx.x) % 134));
//         s93 = X[(b57 + 137)];
//         t110 = (s93 - (0.041666666666666664*((X[(b57 + 1)] - (4.0*s93)) + X[(b57 + 136)] + X[(b57 + 138)] + X[(b57 + 273)])));
//         s94 = X[(b57 + 18633)];
//         s95 = X[(b57 + 37129)];
//         s96 = X[(b57 + 55625)];
//         P1[(a467 + 73984)] = t110;
//         s97 = ((s94 - (0.041666666666666664*((X[(b57 + 18497)] - (4.0*s94)) + X[(b57 + 18632)] + X[(b57 + 18634)] + X[(b57 + 18769)]))) / t110);
//         P1[(a467 + 91940)] = s97;
//         s98 = ((s95 - (0.041666666666666664*((X[(b57 + 36993)] - (4.0*s95)) + X[(b57 + 37128)] + X[(b57 + 37130)] + X[(b57 + 37265)]))) / t110);
//         P1[(a467 + 109896)] = s98;
//         a468 = (gamma1 - 1.0);
//         a469 = (0.5*a468);
//         P1[(a467 + 127852)] = ((a468*(s96 - (0.041666666666666664*((X[(b57 + 55489)] - (4.0*s96)) + X[(b57 + 55624)] + X[(b57 + 55626)] + X[(b57 + 55761)])))) - ((a469*((s97*s97)*t110)) + (a469*((s98*s98)*t110))));
//     }
// }

// __global__ void ker_code2() {
//     if (((((256*blockIdx.x) + threadIdx.x) < 17956))) {
//         int a693, b111;
//         a693 = ((256*blockIdx.x) + threadIdx.x);
//         b111 = ((136*(a693 / 134)) + (((122*blockIdx.x) + threadIdx.x) % 134));
//         P2[a693] = ((0.041666666666666664*((P1[(b111 + 1)] - (4.0*P1[(b111 + 137)])) + P1[(b111 + 136)] + P1[(b111 + 138)] + P1[(b111 + 273)])) + P1[(a693 + 73984)]);
//         P2[(a693 + 17956)] = ((0.041666666666666664*((P1[(b111 + 18497)] - (4.0*P1[(b111 + 18633)])) + P1[(b111 + 18632)] + P1[(b111 + 18634)] + P1[(b111 + 18769)])) + P1[(a693 + 91940)]);
//         P2[(a693 + 35912)] = ((0.041666666666666664*((P1[(b111 + 36993)] - (4.0*P1[(b111 + 37129)])) + P1[(b111 + 37128)] + P1[(b111 + 37130)] + P1[(b111 + 37265)])) + P1[(a693 + 109896)]);
//         P2[(a693 + 53868)] = ((0.041666666666666664*((P1[(b111 + 55489)] - (4.0*P1[(b111 + 55625)])) + P1[(b111 + 55624)] + P1[(b111 + 55626)] + P1[(b111 + 55761)])) + P1[(a693 + 127852)]);
//     }
// }

// __global__ void ker_code3(double gamma1) {
//     if (((((256*blockIdx.x) + threadIdx.x) < 17554))) {
//         double a1111, s298, s299, s300, s301, s302, s303, s307, 
//                 s308, s309, s310, s311, s312, s313, s314, s315, 
//                 s316, s317, s318, s319, s320, s321, s322, s323, 
//                 s324, s325, s326, s327, s328, t193, t194, t195, 
//                 t196, t203, t204, t205, t206, t207, t208, t209, 
//                 t210, t211, t212, t213, t214, t215, t216, t217, 
//                 t218, t219, t220, t221, t222, t223, t224;
//         int a1081, a1082, a1083, a1084, a1085, a1086, a1087, a1088, 
//                 a1089, a1090, a1091, a1092, a1093, a1094, a1095, a1096, 
//                 a1097, a1098, a1099, a1100, a1101, a1102, a1103, a1104, 
//                 a1105, a1106, a1107, a1108, a1109, a1110, a1112, a1113, 
//                 a1114, a1115, a1116, a1117, a1118, a1119, a1120, a1121, 
//                 a1122, a1123, a1124, a1125, a1126, a1127, a1128, a1129, 
//                 a1130, a1131, a1132, a1133, a1134, a1135, a1136, a1137, 
//                 a1138, a1139;
//         a1109 = ((256*blockIdx.x) + threadIdx.x);
//         a1110 = ((134*(a1109 / 131)) + (((125*blockIdx.x) + threadIdx.x) % 131));
//         t207 = (0.5*((((0.58333333333333337*P2[(a1110 + 17957)]) - (0.083333333333333329*P2[(a1110 + 17956)])) + (0.58333333333333337*P2[(a1110 + 17958)])) - (0.083333333333333329*P2[(a1110 + 17959)])));
//         t208 = (0.5*((((0.58333333333333337*P2[(a1110 + 53869)]) - (0.083333333333333329*P2[(a1110 + 53868)])) + (0.58333333333333337*P2[(a1110 + 53870)])) - (0.083333333333333329*P2[(a1110 + 53871)])));
//         t209 = (0.5*((((0.58333333333333337*P2[(a1110 + 1)]) - (0.083333333333333329*P2[a1110])) + (0.58333333333333337*P2[(a1110 + 2)])) - (0.083333333333333329*P2[(a1110 + 3)])));
//         t210 = (t209 + t209);
//         t211 = (t208 + t208);
//         a1111 = sqrt(gamma1);
//         s313 = ((a1111*sqrt(t211))*sqrt((1 / t210)));
//         s314 = t211;
//         if ((((t207 + t207) > 0))) {
//             if ((((s313 - (t207 + t207)) > 0))) {
//                 t212 = ((s314 + (0.083333333333333329*P2[(a1110 + 53871)])) - (((0.58333333333333337*P2[(a1110 + 53869)]) - (0.083333333333333329*P2[(a1110 + 53868)])) + (0.58333333333333337*P2[(a1110 + 53870)])));
//                 s315 = (s313*s313);
//                 s316 = (1 / s315);
//                 s317 = (t212*s316);
//                 t213 = (s317 + ((((0.58333333333333337*P2[(a1110 + 1)]) - (0.083333333333333329*P2[a1110])) + (0.58333333333333337*P2[(a1110 + 2)])) - (0.083333333333333329*P2[(a1110 + 3)])));
//                 P1[a1109] = t213;
//                 a1112 = (a1109 + 17554);
//                 P1[a1112] = (t207 + t207);
//                 a1113 = (a1109 + 35108);
//                 P1[a1113] = ((((0.58333333333333337*P2[(a1110 + 35913)]) - (0.083333333333333329*P2[(a1110 + 35912)])) + (0.58333333333333337*P2[(a1110 + 35914)])) - (0.083333333333333329*P2[(a1110 + 35915)]));
//                 a1114 = (a1109 + 52662);
//                 P1[a1114] = s314;
//             } else {
//                 P1[a1109] = ((((0.58333333333333337*P2[(a1110 + 1)]) - (0.083333333333333329*P2[a1110])) + (0.58333333333333337*P2[(a1110 + 2)])) - (0.083333333333333329*P2[(a1110 + 3)]));
//                 a1115 = (a1109 + 17554);
//                 P1[a1115] = ((((0.58333333333333337*P2[(a1110 + 17957)]) - (0.083333333333333329*P2[(a1110 + 17956)])) + (0.58333333333333337*P2[(a1110 + 17958)])) - (0.083333333333333329*P2[(a1110 + 17959)]));
//                 a1116 = (a1109 + 35108);
//                 P1[a1116] = ((((0.58333333333333337*P2[(a1110 + 35913)]) - (0.083333333333333329*P2[(a1110 + 35912)])) + (0.58333333333333337*P2[(a1110 + 35914)])) - (0.083333333333333329*P2[(a1110 + 35915)]));
//                 a1117 = (a1109 + 52662);
//                 P1[a1117] = ((((0.58333333333333337*P2[(a1110 + 53869)]) - (0.083333333333333329*P2[(a1110 + 53868)])) + (0.58333333333333337*P2[(a1110 + 53870)])) - (0.083333333333333329*P2[(a1110 + 53871)]));
//             }
//         } else {
//             if ((((s313 + t207 + t207) > 0))) {
//                 t214 = ((s314 + (0.083333333333333329*P2[(a1110 + 53871)])) - (((0.58333333333333337*P2[(a1110 + 53869)]) - (0.083333333333333329*P2[(a1110 + 53868)])) + (0.58333333333333337*P2[(a1110 + 53870)])));
//                 s318 = (s313*s313);
//                 s319 = (1 / s318);
//                 s320 = (t214*s319);
//                 t215 = (s320 + ((((0.58333333333333337*P2[(a1110 + 1)]) - (0.083333333333333329*P2[a1110])) + (0.58333333333333337*P2[(a1110 + 2)])) - (0.083333333333333329*P2[(a1110 + 3)])));
//                 P1[a1109] = t215;
//                 a1118 = (a1109 + 17554);
//                 P1[a1118] = (t207 + t207);
//                 a1119 = (a1109 + 35108);
//                 P1[a1119] = ((((0.58333333333333337*P2[(a1110 + 35913)]) - (0.083333333333333329*P2[(a1110 + 35912)])) + (0.58333333333333337*P2[(a1110 + 35914)])) - (0.083333333333333329*P2[(a1110 + 35915)]));
//                 a1120 = (a1109 + 52662);
//                 P1[a1120] = s314;
//             } else {
//                 P1[a1109] = ((((0.58333333333333337*P2[(a1110 + 1)]) - (0.083333333333333329*P2[a1110])) + (0.58333333333333337*P2[(a1110 + 2)])) - (0.083333333333333329*P2[(a1110 + 3)]));
//                 a1121 = (a1109 + 17554);
//                 P1[a1121] = ((((0.58333333333333337*P2[(a1110 + 17957)]) - (0.083333333333333329*P2[(a1110 + 17956)])) + (0.58333333333333337*P2[(a1110 + 17958)])) - (0.083333333333333329*P2[(a1110 + 17959)]));
//                 a1122 = (a1109 + 35108);
//                 P1[a1122] = ((((0.58333333333333337*P2[(a1110 + 35913)]) - (0.083333333333333329*P2[(a1110 + 35912)])) + (0.58333333333333337*P2[(a1110 + 35914)])) - (0.083333333333333329*P2[(a1110 + 35915)]));
//                 a1123 = (a1109 + 52662);
//                 P1[a1123] = ((((0.58333333333333337*P2[(a1110 + 53869)]) - (0.083333333333333329*P2[(a1110 + 53868)])) + (0.58333333333333337*P2[(a1110 + 53870)])) - (0.083333333333333329*P2[(a1110 + 53871)]));
//             }
//         }
//         t216 = (0.5*((((0.58333333333333337*P2[(a1109 + 36046)]) - (0.083333333333333329*P2[(a1109 + 35912)])) + (0.58333333333333337*P2[(a1109 + 36180)])) - (0.083333333333333329*P2[(a1109 + 36314)])));
//         t217 = (0.5*((((0.58333333333333337*P2[(a1109 + 54002)]) - (0.083333333333333329*P2[(a1109 + 53868)])) + (0.58333333333333337*P2[(a1109 + 54136)])) - (0.083333333333333329*P2[(a1109 + 54270)])));
//         t218 = (0.5*((((0.58333333333333337*P2[(a1109 + 134)]) - (0.083333333333333329*P2[a1109])) + (0.58333333333333337*P2[(a1109 + 268)])) - (0.083333333333333329*P2[(a1109 + 402)])));
//         t219 = (t218 + t218);
//         t220 = (t217 + t217);
//         s321 = ((a1111*sqrt(t220))*sqrt((1 / t219)));
//         s322 = t220;
//         if ((((t216 + t216) > 0))) {
//             if ((((s321 - (t216 + t216)) > 0))) {
//                 t221 = ((s322 + (0.083333333333333329*P2[(a1109 + 54270)])) - (((0.58333333333333337*P2[(a1109 + 54002)]) - (0.083333333333333329*P2[(a1109 + 53868)])) + (0.58333333333333337*P2[(a1109 + 54136)])));
//                 s323 = (s321*s321);
//                 s324 = (1 / s323);
//                 s325 = (t221*s324);
//                 t222 = (s325 + ((((0.58333333333333337*P2[(a1109 + 134)]) - (0.083333333333333329*P2[a1109])) + (0.58333333333333337*P2[(a1109 + 268)])) - (0.083333333333333329*P2[(a1109 + 402)])));
//                 a1124 = (a1109 + 70216);
//                 P1[a1124] = t222;
//                 a1125 = (a1109 + 87770);
//                 P1[a1125] = ((((0.58333333333333337*P2[(a1109 + 18090)]) - (0.083333333333333329*P2[(a1109 + 17956)])) + (0.58333333333333337*P2[(a1109 + 18224)])) - (0.083333333333333329*P2[(a1109 + 18358)]));
//                 a1126 = (a1109 + 105324);
//                 P1[a1126] = (t216 + t216);
//                 a1127 = (a1109 + 122878);
//                 P1[a1127] = s322;
//             } else {
//                 a1128 = (a1109 + 70216);
//                 P1[a1128] = ((((0.58333333333333337*P2[(a1109 + 134)]) - (0.083333333333333329*P2[a1109])) + (0.58333333333333337*P2[(a1109 + 268)])) - (0.083333333333333329*P2[(a1109 + 402)]));
//                 a1129 = (a1109 + 87770);
//                 P1[a1129] = ((((0.58333333333333337*P2[(a1109 + 18090)]) - (0.083333333333333329*P2[(a1109 + 17956)])) + (0.58333333333333337*P2[(a1109 + 18224)])) - (0.083333333333333329*P2[(a1109 + 18358)]));
//                 a1130 = (a1109 + 105324);
//                 P1[a1130] = ((((0.58333333333333337*P2[(a1109 + 36046)]) - (0.083333333333333329*P2[(a1109 + 35912)])) + (0.58333333333333337*P2[(a1109 + 36180)])) - (0.083333333333333329*P2[(a1109 + 36314)]));
//                 a1131 = (a1109 + 122878);
//                 P1[a1131] = ((((0.58333333333333337*P2[(a1109 + 54002)]) - (0.083333333333333329*P2[(a1109 + 53868)])) + (0.58333333333333337*P2[(a1109 + 54136)])) - (0.083333333333333329*P2[(a1109 + 54270)]));
//             }
//         } else {
//             if ((((s321 + t216 + t216) > 0))) {
//                 t223 = ((s322 + (0.083333333333333329*P2[(a1109 + 54270)])) - (((0.58333333333333337*P2[(a1109 + 54002)]) - (0.083333333333333329*P2[(a1109 + 53868)])) + (0.58333333333333337*P2[(a1109 + 54136)])));
//                 s326 = (s321*s321);
//                 s327 = (1 / s326);
//                 s328 = (t223*s327);
//                 t224 = (s328 + ((((0.58333333333333337*P2[(a1109 + 134)]) - (0.083333333333333329*P2[a1109])) + (0.58333333333333337*P2[(a1109 + 268)])) - (0.083333333333333329*P2[(a1109 + 402)])));
//                 a1132 = (a1109 + 70216);
//                 P1[a1132] = t224;
//                 a1133 = (a1109 + 87770);
//                 P1[a1133] = ((((0.58333333333333337*P2[(a1109 + 18090)]) - (0.083333333333333329*P2[(a1109 + 17956)])) + (0.58333333333333337*P2[(a1109 + 18224)])) - (0.083333333333333329*P2[(a1109 + 18358)]));
//                 a1134 = (a1109 + 105324);
//                 P1[a1134] = (t216 + t216);
//                 a1135 = (a1109 + 122878);
//                 P1[a1135] = s322;
//             } else {
//                 a1136 = (a1109 + 70216);
//                 P1[a1136] = ((((0.58333333333333337*P2[(a1109 + 134)]) - (0.083333333333333329*P2[a1109])) + (0.58333333333333337*P2[(a1109 + 268)])) - (0.083333333333333329*P2[(a1109 + 402)]));
//                 a1137 = (a1109 + 87770);
//                 P1[a1137] = ((((0.58333333333333337*P2[(a1109 + 18090)]) - (0.083333333333333329*P2[(a1109 + 17956)])) + (0.58333333333333337*P2[(a1109 + 18224)])) - (0.083333333333333329*P2[(a1109 + 18358)]));
//                 a1138 = (a1109 + 105324);
//                 P1[a1138] = ((((0.58333333333333337*P2[(a1109 + 36046)]) - (0.083333333333333329*P2[(a1109 + 35912)])) + (0.58333333333333337*P2[(a1109 + 36180)])) - (0.083333333333333329*P2[(a1109 + 36314)]));
//                 a1139 = (a1109 + 122878);
//                 P1[a1139] = ((((0.58333333333333337*P2[(a1109 + 54002)]) - (0.083333333333333329*P2[(a1109 + 53868)])) + (0.58333333333333337*P2[(a1109 + 54136)])) - (0.083333333333333329*P2[(a1109 + 54270)]));
//             }
//         }
//     }
// }

// __global__ void ker_code4(double gamma1) {
//     if (((((256*blockIdx.x) + threadIdx.x) < 17554))) {
//         double a1208, s385, s387, s388, s389, s390, s391, s392, 
//                 s393;
//         int a1204, a1205, a1206, a1207, a1209, a1210, a1211, a1212;
//         a1204 = ((256*blockIdx.x) + threadIdx.x);
//         a1205 = (a1204 + 17554);
//         s385 = P1[a1205];
//         s387 = (P1[a1204]*s385);
//         P2[a1204] = s387;
//         a1206 = (a1204 + 52662);
//         s388 = P1[a1206];
//         P2[a1205] = ((s387*s385) + s388);
//         a1207 = (a1204 + 35108);
//         s389 = P1[a1207];
//         P2[a1207] = (s387*s389);
//         a1208 = (gamma1 / (gamma1 - 1.0));
//         P2[a1206] = ((a1208*(s385*s388)) + (s387*(((0.5*s385)*s385) + ((0.5*s389)*s389))));
//         a1209 = (a1204 + 70216);
//         s390 = P1[a1209];
//         a1210 = (a1204 + 105324);
//         s391 = P1[a1210];
//         s392 = (s390*s391);
//         P2[a1209] = s392;
//         a1211 = (a1204 + 87770);
//         P2[a1211] = (s392*P1[a1211]);
//         a1212 = (a1204 + 122878);
//         s393 = P1[a1212];
//         P2[a1210] = ((s392*s391) + s393);
//         P2[a1212] = ((a1208*(s391*s393)) + (s392*(((0.5*s390)*s390) + ((0.5*s391)*s391))));
//     }
// }

// __global__ void ker_code5(double gamma1) {
//     if (((((256*blockIdx.x) + threadIdx.x) < 17292))) {
//         double a1428, s500, s501, s502, s503, s504, s505, s506, 
//                 s507, s508, s509, t287, t288, t289, t290, t291, 
//                 t292;
//         int a1427, b265;
//         a1427 = ((256*blockIdx.x) + threadIdx.x);
//         s500 = P1[(a1427 + 131)];
//         s501 = P1[(a1427 + 17685)];
//         t287 = (s501 - (0.041666666666666664*((P1[(a1427 + 17554)] - (2.0*s501)) + P1[(a1427 + 17816)])));
//         s502 = P1[(a1427 + 35239)];
//         t288 = (s502 - (0.041666666666666664*((P1[(a1427 + 35108)] - (2.0*s502)) + P1[(a1427 + 35370)])));
//         s503 = P1[(a1427 + 52793)];
//         t289 = (s503 - (0.041666666666666664*((P1[(a1427 + 52662)] - (2.0*s503)) + P1[(a1427 + 52924)])));
//         s504 = ((s500 - (0.041666666666666664*((P1[a1427] - (2.0*s500)) + P1[(a1427 + 262)])))*t287);
//         P2[(a1427 + 140432)] = s504;
//         P2[(a1427 + 157724)] = ((s504*t287) + t289);
//         P2[(a1427 + 175016)] = (s504*t288);
//         a1428 = (gamma1 / (gamma1 - 1.0));
//         P2[(a1427 + 192308)] = ((a1428*(t287*t289)) + (s504*(((0.5*t287)*t287) + ((0.5*t288)*t288))));
//         b265 = ((134*(a1427 / 132)) + (((124*blockIdx.x) + threadIdx.x) % 132));
//         s505 = P1[(b265 + 70217)];
//         s506 = P1[(b265 + 87771)];
//         t290 = (s506 - (0.041666666666666664*((P1[(b265 + 87770)] - (2.0*s506)) + P1[(b265 + 87772)])));
//         s507 = P1[(b265 + 105325)];
//         t291 = (s507 - (0.041666666666666664*((P1[(b265 + 105324)] - (2.0*s507)) + P1[(b265 + 105326)])));
//         s508 = P1[(b265 + 122879)];
//         t292 = (s508 - (0.041666666666666664*((P1[(b265 + 122878)] - (2.0*s508)) + P1[(b265 + 122880)])));
//         s509 = ((s505 - (0.041666666666666664*((P1[(b265 + 70216)] - (2.0*s505)) + P1[(b265 + 70218)])))*t291);
//         P2[(a1427 + 209600)] = s509;
//         P2[(a1427 + 226892)] = (s509*t290);
//         P2[(a1427 + 244184)] = ((s509*t291) + t292);
//         P2[(a1427 + 261476)] = ((a1428*(t291*t292)) + (s509*(((0.5*t290)*t290) + ((0.5*t291)*t291))));
//     }
// }

// __global__ void ker_code6() {
//     if (((((256*blockIdx.x) + threadIdx.x) < 17292))) {
//         int a1651, b322;
//         a1651 = ((256*blockIdx.x) + threadIdx.x);
//         P1[a1651] = ((0.041666666666666664*((P2[a1651] - (2.0*P2[(a1651 + 131)])) + P2[(a1651 + 262)])) + P2[(a1651 + 140432)]);
//         P1[(a1651 + 17292)] = ((0.041666666666666664*((P2[(a1651 + 17554)] - (2.0*P2[(a1651 + 17685)])) + P2[(a1651 + 17816)])) + P2[(a1651 + 157724)]);
//         P1[(a1651 + 34584)] = ((0.041666666666666664*((P2[(a1651 + 35108)] - (2.0*P2[(a1651 + 35239)])) + P2[(a1651 + 35370)])) + P2[(a1651 + 175016)]);
//         P1[(a1651 + 51876)] = ((0.041666666666666664*((P2[(a1651 + 52662)] - (2.0*P2[(a1651 + 52793)])) + P2[(a1651 + 52924)])) + P2[(a1651 + 192308)]);
//         b322 = ((134*(a1651 / 132)) + (((124*blockIdx.x) + threadIdx.x) % 132));
//         P1[(a1651 + 69168)] = ((0.041666666666666664*((P2[(b322 + 70216)] - (2.0*P2[(b322 + 70217)])) + P2[(b322 + 70218)])) + P2[(a1651 + 209600)]);
//         P1[(a1651 + 86460)] = ((0.041666666666666664*((P2[(b322 + 87770)] - (2.0*P2[(b322 + 87771)])) + P2[(b322 + 87772)])) + P2[(a1651 + 226892)]);
//         P1[(a1651 + 103752)] = ((0.041666666666666664*((P2[(b322 + 105324)] - (2.0*P2[(b322 + 105325)])) + P2[(b322 + 105326)])) + P2[(a1651 + 244184)]);
//         P1[(a1651 + 121044)] = ((0.041666666666666664*((P2[(b322 + 122878)] - (2.0*P2[(b322 + 122879)])) + P2[(b322 + 122880)])) + P2[(a1651 + 261476)]);
//     }
// }

// __global__ void ker_code7() {
//     if (((((256*blockIdx.x) + threadIdx.x) < 17160))) {
//         int a1796, a1797, a1798, b350, b351;
//         a1796 = (threadIdx.x + (256*blockIdx.x));
//         a1797 = (a1796 / 128);
//         a1798 = (threadIdx.x % 128);
//         b350 = ((131*a1797) + a1798);
//         P2[a1796] = (P1[(b350 + 264)] - P1[(b350 + 263)]);
//         P2[(a1796 + 16384)] = (P1[(b350 + 17556)] - P1[(b350 + 17555)]);
//         P2[(a1796 + 32768)] = (P1[(b350 + 34848)] - P1[(b350 + 34847)]);
//         P2[(a1796 + 49152)] = (P1[(b350 + 52140)] - P1[(b350 + 52139)]);
//         b351 = ((132*a1797) + a1798);
//         P2[(a1796 + 65536)] = (P1[(b351 + 69434)] - P1[(b351 + 69302)]);
//         P2[(a1796 + 81920)] = (P1[(b351 + 86726)] - P1[(b351 + 86594)]);
//         P2[(a1796 + 98304)] = (P1[(b351 + 104018)] - P1[(b351 + 103886)]);
//         P2[(a1796 + 114688)] = (P1[(b351 + 121310)] - P1[(b351 + 121178)]);
//     }
// }

// __global__ void ker_code8(double *Y, double a_scale1, double dx1) {
//     if (((((256*blockIdx.x) + threadIdx.x) < 16384))) {
//         double a1847;
//         int a1846, a1848, a1849, a1850;
//         a1846 = (threadIdx.x + (256*blockIdx.x));
//         a1847 = (-(a_scale1) / dx1);
//         Y[a1846] = (a1847*(P2[a1846] + P2[(a1846 + 65536)]));
//         a1848 = (a1846 + 16384);
//         Y[a1848] = (a1847*(P2[a1848] + P2[(a1846 + 81920)]));
//         a1849 = (a1846 + 32768);
//         Y[a1849] = (a1847*(P2[a1849] + P2[(a1846 + 98304)]));
//         a1850 = (a1846 + 49152);
//         Y[a1850] = (a1847*(P2[a1850] + P2[(a1846 + 114688)]));
//     }
// }

// // void full_euler_gpu_hip(double *Y, const double *X, double gamma1, double a_scale1, double dx1) {
// //     dim3 b362(256, 1, 1), b363(256, 1, 1), b364(256, 1, 1), b365(256, 1, 1), b366(256, 1, 1), b367(256, 1, 1), b368(256, 1, 1), b369(256, 1, 1), 
// //     b370(256, 1, 1), g10(65, 1, 1), g2(73, 1, 1), g3(71, 1, 1), g4(71, 1, 1), g5(69, 1, 1), g6(69, 1, 1), g7(68, 1, 1), 
// //     g8(68, 1, 1), g9(68, 1, 1);
// //     hipLaunchKernelGGL(ker_code0, dim3(g2), dim3(b362), 0, 0, X, gamma1);
// //     hipLaunchKernelGGL(ker_code1, dim3(g3), dim3(b363), 0, 0, X, gamma1);
// //     hipLaunchKernelGGL(ker_code2, dim3(g4), dim3(b364), 0, 0);
// //     hipLaunchKernelGGL(ker_code3, dim3(g5), dim3(b365), 0, 0, gamma1);
// //     hipLaunchKernelGGL(ker_code4, dim3(g6), dim3(b366), 0, 0, gamma1);
// //     hipLaunchKernelGGL(ker_code5, dim3(g7), dim3(b367), 0, 0, gamma1);
// //     hipLaunchKernelGGL(ker_code6, dim3(g8), dim3(b368), 0, 0);
// //     hipLaunchKernelGGL(ker_code7, dim3(g9), dim3(b369), 0, 0);
// //     hipLaunchKernelGGL(ker_code8, dim3(g10), dim3(b370), 0, 0, Y, a_scale1, dx1);
// // }

// void full_euler_gpu_cuda(double *Y, const double *X, double gamma1, double a_scale1, double dx1) {
//     dim3 b362(256, 1, 1), b363(256, 1, 1), b364(256, 1, 1), b365(256, 1, 1), b366(256, 1, 1), b367(256, 1, 1), b368(256, 1, 1), b369(256, 1, 1), 
//     b370(256, 1, 1), g10(65, 1, 1), g2(73, 1, 1), g3(71, 1, 1), g4(71, 1, 1), g5(69, 1, 1), g6(69, 1, 1), g7(68, 1, 1), 
//     g8(68, 1, 1), g9(68, 1, 1);
//     ker_code0<<<g2, b362>>>(X,gamma1);
//     ker_code1<<<g3, b363>>>(X,gamma1);
//     ker_code2<<<g4, b364>>>();
//     ker_code3<<<g5, b365>>>(gamma1);
//     ker_code4<<<g6, b366>>>(gamma1);
//     ker_code5<<<g7, b367>>>(gamma1);
//     ker_code6<<<g8, b368>>>();
//     ker_code7<<<g9, b369>>>();
//     ker_code8<<<g9, b370>>>(Y,a_scale1, dx1);
// }


// void destroy_full_euler_gpu() {
// }
using namespace Proto;
typedef BoxData<double> Scalar;
typedef BoxData<double, NUMCOMPS> Vector;

//State: [rho, G0, G1, ..., E]
// Gi = rho*vi
// E = p/(gamma-1) + 0.5*rho*|v|^2
//template<typename T, MemType MEM>
PROTO_KERNEL_START
void
f_thresholdF(
             Var<short>& a_tags,
             Var<double, NUMCOMPS>& a_U)
{
  double thresh = 1.001;
  if (a_U(0) > thresh) {a_tags(0) = 1;}
  else {a_tags(0) = 0;};
};
PROTO_KERNEL_END(f_thresholdF, f_threshold);
template<typename T, MemType MEM>
PROTO_KERNEL_START
void f_consToPrim_(
        Var<T, NUMCOMPS, MEM>&          a_W, 
        const Var<T, NUMCOMPS, MEM>&    a_U,
        double                          a_gamma)
{
    double rho = a_U(0);
    double v2 = 0.0;
    a_W(0) = rho;

    for (int i = 1; i <= DIM; i++)
    {
        double v;
        v = a_U(i) / rho;

        a_W(i) = v;
        v2 += v*v;
    }

    a_W(NUMCOMPS-1) = (a_U(NUMCOMPS-1) - .5 * rho * v2) * (a_gamma - 1.0);
    
}
PROTO_KERNEL_END(f_consToPrim_, f_consToPrim)

template<typename T, MemType MEM>
PROTO_KERNEL_START
void f_upwindState_(
        Var<T, NUMCOMPS, MEM>&       a_out,
        const Var<T, NUMCOMPS, MEM>& a_low,
        const Var<T, NUMCOMPS, MEM>& a_high,
        int                          a_dir,
        double                       a_gamma)
{
    const double& rhol = a_low(0);
    const double& rhor = a_high(0);
    const double& ul = a_low(a_dir+1);
    const double& ur = a_high(a_dir+1);
    const double& pl = a_low(NUMCOMPS-1);
    const double& pr = a_high(NUMCOMPS-1);
    double gamma = a_gamma;
    double rhobar = (rhol + rhor)*.5;
    double pbar = (pl + pr)*.5;
    double ubar = (ul + ur)*.5;
    double cbar = sqrt(gamma*pbar/rhobar);
    double pstar = (pl + pr)*.5 + rhobar*cbar*(ul - ur)*.5;
    double ustar = (ul + ur)*.5 + (pl - pr)/(2*rhobar*cbar);
    int sign;
    if (ustar > 0) 
    {
        sign = -1;
        for (int icomp = 0;icomp < NUMCOMPS;icomp++)
        {
            a_out(icomp) = a_low(icomp);
        }
    }
    else
    {
        sign = 1;
        for (int icomp = 0;icomp < NUMCOMPS;icomp++)
        {
            a_out(icomp) = a_high(icomp);
        }
    }

    double outval = a_out(0) + (pstar - a_out(NUMCOMPS-1))/(cbar*cbar);
    if (cbar + sign*ubar > 0)
    {
        a_out(0) = outval;
        a_out(a_dir+1) = ustar;
        a_out(NUMCOMPS-1) = pstar;
    }
}
PROTO_KERNEL_END(f_upwindState_, f_upwindState)
    
template<typename T, MemType MEM>
PROTO_KERNEL_START
void f_getFlux_(
        Var<T, NUMCOMPS, MEM>&       a_F,
        const Var<T, NUMCOMPS, MEM>& a_W, 
        int                          a_dir,
        double                       a_gamma)
{
    double F0 = a_W(a_dir+1)*a_W(0);
    double W2 = 0.0;
    double gamma = a_gamma;

    a_F(0) = F0;

    for (int d = 1; d <= DIM; d++)
    {
        double Wd = a_W(d);

        a_F(d) = Wd*F0;
        W2 += Wd*Wd;
    }

    a_F(a_dir+1) += a_W(NUMCOMPS-1);
    a_F(NUMCOMPS-1) = gamma/(gamma - 1.0) * a_W(a_dir+1) * a_W(NUMCOMPS-1) + 0.5 * F0 * W2;
    for (int c = 0 ; c < NUMCOMPS; c++)
      {
        a_F(c) = -a_F(c);
      }
}
PROTO_KERNEL_END(f_getFlux_, f_getFlux)

template<typename T, MemType MEM>
PROTO_KERNEL_START
void f_waveSpeedBound_(Var<double,1>& a_speed,
        const Var<T, NUMCOMPS, MEM>& a_W,
        double       a_gamma)
{
    a_speed(0) = DIM*sqrt(a_gamma*a_W(NUMCOMPS-1)/a_W(0));
    for (int dir = 1 ; dir <= DIM; dir++)
    {
      a_speed(0) += fabs(a_W(dir));
    }
}
PROTO_KERNEL_END(f_waveSpeedBound_, f_waveSpeedBound)

template<typename T, MemType MEM = MEMTYPE_DEFAULT>
class BoxOp_Euler : public BoxOp<T, NUMCOMPS, 1, MEM>
{
    public:
    using BoxOp<T,NUMCOMPS,1,MEM>::BoxOp;

    T gamma = 1.4;
    mutable T umax;

    // How many ghost cells does the operator need from the state variables
    inline static Point ghost() { return Point::Ones(4);}
    
    // How many ghost cells does the operator need from the auxiliary variables
    inline static Point auxGhost() { return Point::Zeros();}
    
    // What is the intended order of accuracy of the operator
    inline static constexpr int order() { return 4; }
    
    // Initialization
    inline void init()
    {
        for (int dir = 0; dir < DIM; dir++)
        {
            m_interp_H[dir] = Stencil<double>::CellToFaceH(dir);
            m_interp_L[dir] = Stencil<double>::CellToFaceL(dir);
            m_divergence[dir] = Stencil<double>::FluxDivergence(dir);
            m_laplacian_f[dir] = Stencil<double>::LaplacianFace(dir);
        }
    }

    // Helper Function
    inline void computeFlux(
            BoxData<T, NUMCOMPS>& a_flux,
            const BoxData<T, NUMCOMPS>& a_W_ave,
            int a_dir) const
    {
        PR_TIME("BoxOp_Euler::computeFlux");
        Vector W_ave_L = m_interp_L[a_dir](a_W_ave); 
        Vector W_ave_H = m_interp_H[a_dir](a_W_ave); 
        Vector W_ave_f = forall<double,NUMCOMPS>(f_upwindState, W_ave_L, W_ave_H, a_dir, gamma);
#if DIM>1
        Vector F_bar_f = forall<double,NUMCOMPS>(f_getFlux, W_ave_f, a_dir,  gamma);
        Vector W_f = Operator::deconvolveFace(W_ave_f, a_dir);
#else
        Vector W_f = W_ave_f;
#endif
        a_flux = forall<double,NUMCOMPS>(f_getFlux, W_f, a_dir, gamma);
#if DIM>1
        // a_flux += m_laplacian_f[a_dir](F_bar_f, 1.0/24.0);
#endif
    }
   
    // Flux Definition
    inline void flux(
            BoxData<T, NUMCOMPS>& a_flux,
            const BoxData<T, NUMCOMPS>& a_U,
            int a_dir) const
    {
        PR_TIME("BoxOp_Euler::flux");
        
        
        Vector W_bar = forall<double, NUMCOMPS>(f_consToPrim, a_U, gamma);
        Vector U = Operator::deconvolve(a_U);
        Vector W = forall<double, NUMCOMPS>(f_consToPrim, U, gamma);
        Vector W_ave = Operator::_convolve(W, W_bar);
        computeFlux(a_flux, W_ave, a_dir);
    }
    // Apply BCs by filling ghost cells in stage values. For Euler, this is done by calling
    // exchange. For the MHD code, it will be more complicated.
    // The interface is very provisional. We expect it to evolve as we d more real problems.
    inline void bcStage(
                        LevelBoxData<T,NUMCOMPS>& a_UStage,
                        const LevelBoxData<T,NUMCOMPS>& a_U0,
                        int a_stage)
    {
      a_UStage.exchange();
    }                 
    
    // Apply Operator
    inline void operator()(
            BoxData<T, NUMCOMPS>&                   a_Rhs,
            Array<BoxData<T, NUMCOMPS>, DIM>&  a_fluxes,
            const BoxData<T, NUMCOMPS>&             a_U,
            T                                       a_scale = 1.0) const
    {
        T dx = this->dx()[0];
        PR_TIME("BoxOp_Euler::operator()");        
        // COMPUTE W_AVE
        std::cout << "begin computation" << std::endl;
        a_Rhs.setVal(0.0);

        // auto start = std::chrono::high_resolution_clock::now();
        // full_euler_gpu_cuda(a_Rhs.data(), a_U.data(), gamma,  a_scale, dx);
        // auto stop = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        // std::cout << duration.count() << " us" <<std::endl;
        // exit(0);   
        Vector W_bar = forall<double, NUMCOMPS>(f_consToPrim, a_U, gamma);
        // Vector U = Operator::deconvolve(a_U);
        // Vector W = forall<double, NUMCOMPS>(f_consToPrim, U, gamma);
        // Vector W_ave = Operator::_convolve(W, W_bar);
        
        // // // COMPUTE MAX WAVE SPEED
        // Box rangeBox = a_U.box().grow(-ghost());
        // Scalar uabs = forall<double>(f_waveSpeedBound, rangeBox, W, gamma);
        // umax = uabs.absMax();

        // // // COMPUTE DIV FLUXES
        // for (int dir = 0; dir < DIM; dir++)
        // {
        //     computeFlux(a_fluxes[dir], W_ave, dir);
        //     // a_Rhs += m_divergence[dir](a_fluxes[dir]);
        // }
        // a_Rhs *= (a_scale / dx); //Assuming isotropic grid spacing
        std::cout << "end computation" << std::endl;
    }
#ifdef PR_AMR
  static inline void generateTags(
                                  TagData& a_tags,
                                  BoxData<T, NUMCOMPS>& a_state)
  {
    forallInPlace(f_threshold, a_tags, a_state);
  }
#endif
private:

  //Array<std::shared_ptr<Array<BoxData<double>,DIM>>,DIM> m_data;
  //std::shared_ptr<BoxData<double, 1>> m_data;
  BoxData<double> m_data;
  Stencil<T> m_interp_H[DIM];
  Stencil<T> m_interp_L[DIM];
  Stencil<T> m_divergence[DIM];
  Stencil<T> m_laplacian_f[DIM];
};

#endif //end include guard