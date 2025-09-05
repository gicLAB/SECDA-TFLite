void ACCNAME::ReadBias() {
    // #pragma HLS pipeline II = 1
    res[a + 0][b] = din1->read().data;
    res[a + 1][b] = din2->read().data;
    res[a + 2][b] = din3->read().data;
    res[a + 3][b] = din4->read().data;
}